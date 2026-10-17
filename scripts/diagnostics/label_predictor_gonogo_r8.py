"""Go/no-go for supervised adaptive sampling: can a network predict the per-slice oracle labels of
oracle_labels_r8.py from the sampler's input well enough to beat the best non-adaptive maps on held-out slices?

The predictor is the baseline conv sampler's U-Net, fine-tuned (reconstructor untouched) to the training
labels' probability maps with a BCE over the available k-space; 10% of the labelled training slices pick the
epoch. Every arm is scored on the whole validation set with the frozen baseline VarNet, through the trainer's
masked PSNR, over the same seeded Bernoulli draws (common random numbers across arms):

  native               the baseline conv sampler
  prior                the shared map every label started from
  mean_label           logit of the training labels' mean probability (a population refit, non-adaptive)
  anatomy_mean_label   the same per anatomy (the strongest non-adaptive control)
  fitted_anatomy_map   one score map per anatomy fitted through the VarNet on the labelled training slices
                       with the labels' estimator (the population refit)
  knn                  mean label of the k nearest training slices of the same anatomy, by centre k-space magnitude
  predictor            the fine-tuned network
  oracle               the slice's own label (labelled validation slices only; the headroom)

Go: predictor or knn beats the best of the non-adaptive arms (native, prior, mean_label, anatomy_mean_label,
fitted_anatomy_map) by a margin whose volume-bootstrap 95% interval excludes 0.
Label predictability is also read directly as the correlation, over available locations, of an arm's deviation
from anatomy_mean_label with the label's own deviation (on labelled validation and training slices). Its ceiling
is the same correlation between two labels of one slice fitted with different seeds (`--retest-dir`).

    python scripts/diagnostics/label_predictor_gonogo_r8.py
"""
import argparse
import copy
import json
from pathlib import Path

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
from omegaconf import OmegaConf

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.training.builders import build_dataloaders
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img
from oracle_labels_r8 import (OUT_DIR, PROBABILITY_EPS, RUN, load_frozen, loader_over, masks_from_logits,
                              recon_loss, sampler_input, seeded_psnr, to_device)

OUT = Path('reports/conv_vs_loupe_r8/label_gonogo.json')
ANATOMIES = ('brain', 'knee')
UNCLAMPED_LIMIT = 1e4
CENTRE_CROP = 16  # half-width of the centre k-space window used as the knn feature
BOOTSTRAPS = 2000
CONTROLS = ('native', 'prior', 'mean_label', 'anatomy_mean_label', 'fitted_anatomy_map')
ADAPTIVE = ('knn', 'predictor')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--label-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight', choices=('none', 'gain'), default='none',
                        help="'gain' weights each slice by its label's PSNR gain over the prior (clipped at 0)")
    parser.add_argument('--deviation-weight', type=float, default=0.0,
                        help='adds this times (1 - per-slice correlation of the predicted and label deviations from '
                             'the anatomy mean), a scale-free term the shared map cannot satisfy')
    parser.add_argument('--screen', action='store_true',
                        help='fit the predictor and report predictability only (no fitted maps, no PSNR evaluation)')
    parser.add_argument('--predictor-input', choices=('observed', 'full', 'centre'), default='observed',
                        help="'full' feeds the predictor the fully sampled SENSE-combined k-space instead of the "
                             "observed 10x10 centre: whether the labels are learnable at all (privileged, --screen only); "
                             "'centre' feeds a --centre-size x --centre-size centre block, as a larger calibration scan "
                             "would (--screen only: the labels and reconstructor assume the 10x10 centre)")
    parser.add_argument('--centre-size', type=int, default=24)
    parser.add_argument('--knn', type=int, default=16)
    parser.add_argument('--shared-steps', type=int, default=300, help='steps of each fitted per-anatomy map')
    parser.add_argument('--shared-lr', type=float, default=0.02)
    parser.add_argument('--retest-dir', type=Path, default=Path('/tmp/bkadota_oracle_labels_r8_retest'),
                        help='validation labels refitted with another seed, for the label noise ceiling')
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--val-limit', type=int, default=None, help='score only the first n validation slices (smoke tests)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_shards(label_dir, split):
    shards = [torch.load(path) for path in sorted(label_dir.glob(f'{split}_*.pt'))]
    if not shards:
        raise FileNotFoundError(f"no {split} label shards in {label_dir}")
    merged = {key: torch.cat([s[key] for s in shards]) for key in shards[0] if torch.is_tensor(shards[0][key])}
    merged['anatomy'] = np.array([a for s in shards for a in s['anatomy']])
    return merged


def label_probability(labels):
    return scores_to_prob(labels['label_logits'].float())


def available_of(labels):
    return (~labels['old_mask'] & labels['valid_mask']).float()


def logit_of(probability):
    return torch.logit(probability.clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS))


def knn_feature(kspace_in):
    """(unit-norm centre k-space magnitude, log of its norm): the shape and the overall signal, which tracks support."""
    centre_h, centre_w = kspace_in.shape[1] // 2, kspace_in.shape[2] // 2
    window = kspace_in[:, centre_h - CENTRE_CROP:centre_h + CENTRE_CROP, centre_w - CENTRE_CROP:centre_w + CENTRE_CROP]
    magnitude = window.float().norm(dim=-1).flatten(1)
    norm = magnitude.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return magnitude / norm, norm.log()


def knn_probability(train, train_features, features, anatomies, k):
    """Per query slice, the mean training-label probability of its k nearest same-anatomy training slices
    (similarity = cosine of the centre magnitudes minus the gap in log signal norm)."""
    probability = label_probability(train)
    shape, log_norm = features
    train_shape, train_log_norm = train_features
    out = torch.empty(len(shape), *probability.shape[1:])
    for anatomy in ANATOMIES:
        pool = torch.as_tensor(train['anatomy'] == anatomy)
        queries = torch.as_tensor(anatomies == anatomy)
        if queries.any():
            similarity = shape[queries] @ train_shape[pool].T - (log_norm[queries] - train_log_norm[pool].T).abs()
            out[queries] = probability[pool][similarity.topk(k, dim=1).indices].mean(1)
    return out


class Predictor:
    """The sampler U-Net acting on stored sampler inputs; `unclamped` drops the score clamp so BCE never saturates."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.net = copy.deepcopy(model.sampler.mask_net).requires_grad_(True)
        self.clamped_limit = self.net.logit_limit

    def probability(self, kspace_in, old_mask, unclamped=False):
        self.net.logit_limit = UNCLAMPED_LIMIT if unclamped else self.clamped_limit
        sampler_input = torch.cat([kspace_in, kspace_in * old_mask.permute(0, 2, 3, 1)], dim=-1)
        if self.model.sampler.image_domain_input:
            sampler_input = self.model.sampler._to_image_domain(sampler_input)
        probability = self.net(sampler_input, old_mask)[0]
        self.net.logit_limit = self.clamped_limit
        return probability


@torch.no_grad()
def full_sampler_input(model, loader, indices, device, chunk, centre_size=None):
    """The sampler input of oracle_labels_r8.sampler_input with every k-space location observed [N, H, W, 2], or
    only a centred `centre_size` x `centre_size` block."""
    out = []
    for batch in loader_over(loader, indices, chunk):
        kspace = batch['kspace'].to(device)
        if centre_size is not None:
            height, width = kspace.shape[2:4]
            window = torch.zeros(height, width, device=device)
            top, left = height // 2 - centre_size // 2, width // 2 - centre_size // 2
            window[top:top + centre_size, left:left + centre_size] = 1
            kspace = kspace * window[None, None, :, :, None]
        complex_kspace = torch.view_as_complex(kspace.contiguous())
        sense_maps, _ = model.sensetivity_net(complex_kspace, mask_method=model.sense_mask_method)
        combined = (ifft_2d_img(complex_kspace) * sense_maps).sum(dim=1)
        out.append(torch.view_as_real(fft_2d_img(combined)).half().cpu())
    return torch.cat(out)


def batch_of(labels, rows, device):
    return (labels['predictor_input'][rows].float().to(device), labels['old_mask'][rows].float().to(device),
            label_probability(labels)[rows].to(device), available_of(labels)[rows].to(device))


def masked_bce(probability, target, available, weight):
    probability = probability.clamp(PROBABILITY_EPS, 1 - PROBABILITY_EPS)
    bce = -(target * probability.log() + (1 - target) * (1 - probability).log())
    per_slice = (bce * available).flatten(1).sum(1) / available.flatten(1).sum(1)
    return (per_slice * weight).sum() / weight.sum()


def masked_deviation_correlation(probability, target, reference, available):
    """Per slice, the correlation over available locations of (probability - reference) and (target - reference)."""
    def centred(deviation):
        deviation = deviation.flatten(1)
        mean = (deviation * keep).sum(1, keepdim=True) / count
        return (deviation - mean) * keep
    keep = available.flatten(1)
    count = keep.sum(1, keepdim=True)
    x, y = centred(probability - reference), centred(target - reference)
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-12)


def predictor_loss(probability, target, reference, available, weight, deviation_weight):
    loss = masked_bce(probability, target, available, weight)
    if deviation_weight:
        correlation = masked_deviation_correlation(probability, target, reference, available)
        loss = loss + deviation_weight * ((1 - correlation) * weight).sum() / weight.sum()
    return loss


@torch.no_grad()
def predicted_probability(predictor, labels, rows, device, chunk=32):
    out = []
    for part in torch.as_tensor(rows).split(chunk):
        kspace_in, old_mask, _, _ = batch_of(labels, part, device)
        out.append(predictor.probability(kspace_in, old_mask).cpu())
    return torch.cat(out)


@torch.no_grad()
def mean_loss(predictor, labels, reference, rows, device, deviation_weight):
    probability = predicted_probability(predictor, labels, rows, device)
    return float(predictor_loss(probability, label_probability(labels)[rows], reference[rows],
                                available_of(labels)[rows], torch.ones(len(rows)), deviation_weight))


def fit_predictor(predictor, train, reference, args):
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(train['label_logits']))
    early_stop, fit_rows = order[:len(order) // 10], order[len(order) // 10:]
    gain = (train['psnr_label'] - train['psnr_prior']).clamp(min=0)
    weight = gain / gain.mean().clamp(min=1e-6) if args.weight == 'gain' else torch.ones_like(gain)
    optimizer = torch.optim.Adam(predictor.net.parameters(), lr=args.lr)
    def early_stop_loss():
        return mean_loss(predictor, train, reference, early_stop, args.device, args.deviation_weight)

    best = {'epoch': 0, 'loss': early_stop_loss(), 'state': copy.deepcopy(predictor.net.state_dict())}
    trace = {0: best['loss']}
    for epoch in range(1, args.epochs + 1):
        predictor.net.train()
        for rows in torch.as_tensor(rng.permutation(fit_rows)).split(args.batch_size):
            kspace_in, old_mask, target, available = batch_of(train, rows, args.device)
            loss = predictor_loss(predictor.probability(kspace_in, old_mask, unclamped=True), target,
                                  reference[rows].to(args.device), available, weight[rows].to(args.device),
                                  args.deviation_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predictor.net.eval()
        trace[epoch] = early_stop_loss()
        print(f"epoch {epoch}: early-stop loss {trace[epoch]:.5f}", flush=True)
        if trace[epoch] < best['loss']:
            best = {'epoch': epoch, 'loss': trace[epoch], 'state': copy.deepcopy(predictor.net.state_dict())}
    predictor.net.load_state_dict(best['state'])
    return {'early_stop_loss': trace, 'best_epoch': best['epoch'], 'fit_rows': fit_rows, 'early_stop_rows': early_stop}


def fit_anatomy_maps(model, loss_type, train_loader, train, prior, args):
    """One free score map per anatomy, from the prior, optimised through the frozen VarNet on that anatomy's labelled training slices."""
    rng = np.random.RandomState(args.seed)
    maps = {}
    for anatomy in ANATOMIES:
        indices = train['indices'][torch.as_tensor(train['anatomy'] == anatomy)].numpy()
        logits = prior.clone().to(args.device).requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=args.shared_lr)
        step = 0
        while step < args.shared_steps:
            for batch in loader_over(train_loader, rng.permutation(indices), args.chunk):
                part = to_device(batch, args.device)
                mask = masks_from_logits(model, logits.expand(len(part['kspace']), -1, -1, -1), part['kspace'])
                loss = recon_loss(model, loss_type, part, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if step == args.shared_steps:
                    break
        maps[anatomy] = logits.detach().cpu()
        print(f"fitted {anatomy} map ({args.shared_steps} steps)", flush=True)
    return maps


def retest_ceiling(retest_dir, val_labels, anatomy_mean, arms):
    """Deviation correlation between two independently fitted labels of the same slices, and the arms' on those slices."""
    if not retest_dir.exists() or not list(retest_dir.glob('val_*.pt')):
        return None
    retest = load_shards(retest_dir, 'val')
    row_of = {int(index): row for row, index in enumerate(val_labels['indices'])}
    rows = [row_of[int(index)] for index in retest['indices']]
    first = {key: value[rows] for key, value in val_labels.items() if len(value) == len(val_labels['label_logits'])}
    reference = anatomy_reference(anatomy_mean, first['anatomy'])
    label, available = label_probability(first), available_of(first)
    result = {'slices': len(rows),
              'label_vs_retest_label': deviation_correlation(label_probability(retest), reference, label, available)}
    result.update({name: deviation_correlation(probability[rows], reference, label, available)
                   for name, probability in arms.items()})
    return result


def deviation_correlation(probability, reference, label, available):
    """Mean over slices of corr(probability - reference, label - reference) over the available locations."""
    correlations = []
    for p, r, l, a in zip(probability, reference, label, available):
        keep = a.bool()
        x, y = (p - r)[keep].double(), (l - r)[keep].double()
        correlations.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    return float(np.nanmean(correlations))


def anatomy_reference(anatomy_mean, anatomies):
    return torch.stack([anatomy_mean[a] for a in anatomies])


def predictability(arms, labels, anatomy_mean):
    reference = anatomy_reference(anatomy_mean, labels['anatomy'])
    label, available = label_probability(labels), available_of(labels)
    return {name: deviation_correlation(probability, reference, label, available) for name, probability in arms.items()}


@torch.no_grad()
def evaluate(model, predictor, val_loader, logits_of, val_labels, args):
    """Per validation slice and arm, the seeded trainer-path PSNR; `logits_of(name, kspace_in, anatomies)` gives map arms."""
    native_net = model.sampler.mask_net
    labelled = {int(index): row for row, index in enumerate(val_labels['indices'])} if val_labels else {}
    indices = np.arange(len(val_loader.dataset))[:args.val_limit]
    psnr = {name: [] for name in ('native', 'predictor', *CONTROLS[1:], 'knn', 'oracle')}
    anatomies, volumes = [], []
    for chunk_index, batch in enumerate(loader_over(val_loader, indices, args.chunk)):
        part = to_device(batch, args.device)
        kspace_in, _, _ = sampler_input(model, part['kspace'])
        names = np.array(batch['anatomy'])
        rows = indices[chunk_index * args.chunk:chunk_index * args.chunk + len(names)]

        def native_mask():
            return model.propose_mask(part['kspace'], anatomy_labels=part['label'])['mask']

        psnr['native'].append(seeded_psnr(model, part, native_mask, chunk_index))
        model.sampler.mask_net = predictor.net
        psnr['predictor'].append(seeded_psnr(model, part, native_mask, chunk_index))
        model.sampler.mask_net = native_net
        for name in (*CONTROLS[1:], 'knn'):
            logits = logits_of(name, kspace_in.cpu(), names).to(args.device)
            psnr[name].append(seeded_psnr(model, part, lambda: masks_from_logits(model, logits, part['kspace']), chunk_index))
        oracle_logits = logits_of('anatomy_mean_label', kspace_in.cpu(), names).clone()
        has_label = torch.tensor([int(r) in labelled for r in rows])
        for position, row in enumerate(rows):
            if int(row) in labelled:
                oracle_logits[position] = val_labels['label_logits'][labelled[int(row)]].float()
        oracle = seeded_psnr(model, part, lambda: masks_from_logits(model, oracle_logits.to(args.device), part['kspace']), chunk_index)
        psnr['oracle'].append(torch.where(has_label, oracle, torch.full_like(oracle, float('nan'))))
        anatomies += list(names)
        volumes += list(batch['fname'])
        if chunk_index % 10 == 0:
            print(f"evaluated {len(anatomies)}/{len(indices)}", flush=True)
    return {name: torch.cat(values).numpy() for name, values in psnr.items()}, np.array(anatomies), np.array(volumes)


def bootstrap_ci(difference, volumes, rng):
    """Mean of a per-slice difference and its 95% interval, resampling whole volumes."""
    keep = ~np.isnan(difference)
    difference, volumes = difference[keep], volumes[keep]
    if len(difference) == 0:
        return None
    unique = np.unique(volumes)
    by_volume = [difference[volumes == v] for v in unique]
    means = [np.concatenate([by_volume[i] for i in rng.randint(len(unique), size=len(unique))]).mean()
             for _ in range(BOOTSTRAPS)]
    return {'mean': float(difference.mean()), 'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            'slices': int(len(difference)), 'volumes': int(len(unique))}


def summarise(psnr, anatomies, volumes, seed):
    """Adaptive arms against every non-adaptive control; go = beating the best control with the interval above 0."""
    rng = np.random.RandomState(seed)
    labelled = ~np.isnan(psnr['oracle'])
    summary = {'mean_psnr': {name: float(np.nanmean(values)) for name, values in psnr.items() if name != 'oracle'},
               'mean_psnr_by_anatomy': {a: {name: float(np.nanmean(values[anatomies == a])) for name, values in psnr.items()
                                            if name != 'oracle'} for a in ANATOMIES}}
    best_control = max(CONTROLS, key=lambda name: float(np.mean(psnr[name])))
    summary['best_control'] = best_control
    summary['adaptive_vs_controls'] = {
        arm: {control: bootstrap_ci(psnr[arm] - psnr[control], volumes, rng) for control in CONTROLS}
        for arm in ADAPTIVE}
    summary['adaptive_vs_best_control_by_anatomy'] = {
        a: {arm: bootstrap_ci((psnr[arm] - psnr[best_control])[anatomies == a], volumes[anatomies == a], rng)
            for arm in ADAPTIVE} for a in ANATOMIES}
    summary['labelled_subset_vs_best_control'] = {
        arm: bootstrap_ci(np.where(labelled, psnr[arm] - psnr[best_control], np.nan), volumes, rng)
        for arm in ('oracle', *ADAPTIVE)}
    summary['go'] = {arm: summary['adaptive_vs_controls'][arm][best_control]['ci95'][0] > 0 for arm in ADAPTIVE}
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, loss_type = load_frozen(args.device)
    if args.predictor_input != 'observed' and not args.screen:
        raise ValueError(f"--predictor-input {args.predictor_input} is scored with --screen only")
    train = load_shards(args.label_dir, 'train')
    val_labels = load_shards(args.label_dir, 'val')
    cfg = OmegaConf.load(RUN / '.hydra' / 'config.yaml')
    cfg.datasets.num_workers = 4
    train_loader, val_loader = build_dataloaders(cfg)
    for labels, loader in ((train, train_loader), (val_labels, val_loader)):
        centre = args.centre_size if args.predictor_input == 'centre' else None
        labels['predictor_input'] = (labels['kspace_in'] if args.predictor_input == 'observed' else
                                     full_sampler_input(model, loader, labels['indices'].numpy(), args.device, args.chunk, centre))
    prior = torch.load(args.label_dir / 'prior.pt')
    print(f"{len(train['label_logits'])} training labels, {len(val_labels['label_logits'])} validation labels; "
          f"training label - prior {float((train['psnr_label'] - train['psnr_prior']).mean()):+.3f} dB", flush=True)

    train_probability = label_probability(train)
    mean_label = train_probability.mean(0)
    anatomy_mean = {a: train_probability[torch.as_tensor(train['anatomy'] == a)].mean(0) for a in ANATOMIES}
    train_features = knn_feature(train['kspace_in'])

    predictor = Predictor(model, args.device)
    native_val_probability = predicted_probability(predictor, val_labels, np.arange(len(val_labels['label_logits'])), args.device)
    fit = fit_predictor(predictor, train, anatomy_reference(anatomy_mean, train['anatomy']), args)
    predictor.net.eval().requires_grad_(False)

    val_rows = np.arange(len(val_labels['label_logits']))
    train_check = fit['fit_rows'][:128]
    train_subset = {key: value[train_check] for key, value in train.items() if len(value) == len(train['label_logits'])}
    loo_features = knn_feature(val_labels['kspace_in'])
    val_arms = {
        'native': native_val_probability,
        'predictor': predicted_probability(predictor, val_labels, val_rows, args.device),
        'knn': knn_probability(train, train_features, loo_features, val_labels['anatomy'], args.knn),
    }
    predictability_report = {
        'val': predictability(val_arms, val_labels, anatomy_mean),
        'retest_ceiling': retest_ceiling(args.retest_dir, val_labels, anatomy_mean, val_arms),
        'train_fit_rows': predictability({
            'predictor': predicted_probability(predictor, train_subset, np.arange(len(train_check)), args.device),
        }, train_subset, anatomy_mean),
    }
    print('predictability', json.dumps(predictability_report), flush=True)
    fit_report = {'best_epoch': fit['best_epoch'], 'early_stop_loss': fit['early_stop_loss']}
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    predictor_file = args.label_dir / f'predictor_{args.predictor_input}_{args.weight}_lr{args.lr:g}_dev{args.deviation_weight:g}.pt'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.screen:
        args.out.write_text(json.dumps({'predictability': predictability_report, 'fit': fit_report,
                                        'config': config}, indent=2) + '\n')
        torch.save(predictor.net.state_dict(), predictor_file)
        return

    def logits_of(name, kspace_in, anatomies):
        count = len(anatomies)
        if name == 'prior':
            return prior.expand(count, -1, -1, -1)
        if name == 'mean_label':
            return logit_of(mean_label).expand(count, -1, -1, -1)
        if name == 'anatomy_mean_label':
            return logit_of(anatomy_reference(anatomy_mean, anatomies))
        if name == 'fitted_anatomy_map':
            return torch.cat([fitted_maps[a] for a in anatomies])
        if name == 'knn':
            return logit_of(knn_probability(train, train_features, knn_feature(kspace_in), anatomies, args.knn))
        raise ValueError(name)

    fitted_maps = fit_anatomy_maps(model, loss_type, train_loader, train, prior, args)
    psnr, anatomies, volumes = evaluate(model, predictor, val_loader, logits_of, val_labels, args)
    results = {'summary': summarise(psnr, anatomies, volumes, args.seed),
               'predictability': predictability_report,
               'fit': fit_report,
               'labels': {'train': len(train['label_logits']), 'val': len(val_labels['label_logits']),
                          'train_label_minus_prior_db': float((train['psnr_label'] - train['psnr_prior']).mean()),
                          'val_label_minus_prior_db': float((val_labels['psnr_label'] - val_labels['psnr_prior']).mean())},
               'config': config}
    print(json.dumps(results['summary'], indent=2), flush=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')
    torch.save(predictor.net.state_dict(), predictor_file)


if __name__ == '__main__':
    main()
