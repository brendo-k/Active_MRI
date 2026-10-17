"""Toy joint sampler / reconstructor training where the adaptive optimum is known (CPU, about a minute per run).

Images are 32x32 and real: a random smooth blob plus bars of limited extent. Class A has vertical bars (k-space
energy along the kx axis), class B horizontal bars (along ky). With `--purity` p each bar has its class's
orientation with probability p and the other one otherwise, so p = 1 makes the classes maximally different and
p = 0.5 makes them identical. Single coil, Cartesian point sampling at R=8 (128 of 1024 locations, the 4x4 centre
preselected), and the conv sampler's estimator: `scores_to_prob` (logit soft-clamped at +/-6 with leak 0.01, floored
sigmoid), RescaleProbMap, Bernoulli draw with a floored straight-through backward. Without the clamp an MLP sampler's
logits grow without bound and its masks saturate within a few hundred steps.

Samplers:
  static   one logit map (LOUPE)
  table    one logit map per class, given the class (LOUPE with anatomy conditioning)
  net      MLP on the acquired 4x4 centre k-space; its output bias is a per-location map, so it can represent `table`
           exactly if it decodes the class from the centre
Reconstructor: 3 cascades of a 4-layer CNN with data consistency k <- (1 - m) k + m y (differentiable in m).

Stages (`--stage`):
  generalist  reconstructor alone on a broad family of random variable-density masks (row bands, column bands,
              isotropic), saved for later stages
  fit         a sampler (and the reconstructor unless `--freeze-recon`) trained from `--recon-init` and
              `--sampler-init` checkpoints or from scratch

Every `--eval-every` steps: val PSNR, the correlation of the class-mean effective probability maps (1 = static),
the share of each class's non-centre probability in its own axis band, and the label-swap gain (PSNR with the
image's own mask minus with a mask made for an image of the other class), through the current reconstructor.

    python scripts/diagnostics/toy_adaptive_sampling.py --stage generalist --out runs/toy/generalist
    python scripts/diagnostics/toy_adaptive_sampling.py --stage fit --sampler table --recon-init runs/toy/generalist --freeze-recon --out ...
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import _path  # noqa: F401
from activemri.samplers.conv import scores_to_prob
from activemri.utils.mask_binarization import RescaleProbMap, threshold_random_mask

SIZE = 32
CENTRE = 4
BUDGET = SIZE * SIZE // 8
NEW = BUDGET - CENTRE * CENTRE
FLOOR = 0.05
BARS = 5
CENTRE_FEATURES = CENTRE  # the sampler sees acquired data only: a wider window would let the mask leak unacquired k-space
BAND = 2
VAL_IMAGES = 512
EVAL_DRAWS = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('generalist', 'fit', 'probe', 'regress', 'gradients'), required=True)
    parser.add_argument('--draws', type=int, default=16, help='gradients stage: Bernoulli draws per image')
    parser.add_argument('--sampler', choices=('static', 'table', 'net'), default='static')
    parser.add_argument('--recon-init', type=Path, default=None)
    parser.add_argument('--sampler-init', type=Path, default=None, help='a static or table run: its logits start the sampler')
    parser.add_argument('--freeze-recon', action='store_true')
    parser.add_argument('--net-input', choices=NET_INPUTS, default='raw')
    parser.add_argument('--net-init', choices=('zero', 'default'), default='zero',
                        help='zero: the head starts at 0, so the net starts as a static mask')
    parser.add_argument('--purity', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--logit-lr', type=float, default=5e-3)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))


def centre_mask():
    mask = torch.zeros(SIZE, SIZE)
    low = SIZE // 2 - CENTRE // 2
    mask[low:low + CENTRE, low:low + CENTRE] = 1
    return mask


def axis_bands():
    """(row band |ky| <= BAND: where vertical bars (class 0) put their energy, column band |kx| <= BAND: class 1),
    both without the centre."""
    offsets = (torch.arange(SIZE) - SIZE // 2).abs() <= BAND
    rows = offsets[:, None].expand(SIZE, SIZE).float()
    cols = offsets[None, :].expand(SIZE, SIZE).float()
    return rows * (1 - centre_mask()), cols * (1 - centre_mask())


def images(count, purity, generator):
    """[count, SIZE, SIZE] real images and [count] labels (0: vertical bars, 1: horizontal)."""
    def uniform(low, high, *shape):
        return low + (high - low) * torch.rand(*shape, generator=generator)

    grid = torch.arange(SIZE).float()
    labels = (torch.rand(count, generator=generator) < 0.5).long()
    cy, cx, width = uniform(8, 24, count, 1, 1), uniform(8, 24, count, 1, 1), uniform(3, 8, count, 1, 1)
    image = uniform(0.5, 1.0, count, 1, 1) * torch.exp(
        -((grid[None, :, None] - cy) ** 2 + (grid[None, None, :] - cx) ** 2) / (2 * width ** 2))
    for _ in range(BARS):
        centre = uniform(2, SIZE - 2, count, 1)
        half_width = torch.randint(1, 4, (count, 1), generator=generator).float() / 2 + 0.25
        start = uniform(0, 12, count, 1)
        stop = start + uniform(12, 20, count, 1)
        amplitude = uniform(0.3, 1.0, count, 1) * torch.where(torch.rand(count, 1, generator=generator) < 0.5, -1.0, 1.0)
        across = (torch.sigmoid(4 * (grid + 0.5 - (centre - half_width))) - torch.sigmoid(4 * (grid - 0.5 - (centre + half_width))))
        along = torch.sigmoid(2 * (grid - start)) * torch.sigmoid(2 * (stop - grid))
        bar = amplitude[:, :, None] * along[:, :, None] * across[:, None, :]  # varies along x: vertical bar
        own = torch.rand(count, generator=generator) < purity
        horizontal = (labels == 1) == own
        image = image + torch.where(horizontal[:, None, None], bar.transpose(1, 2), bar)
    return image, labels


class Recon(nn.Module):
    def __init__(self, cascades=3, chans=32):
        super().__init__()

        def block():
            return nn.Sequential(nn.Conv2d(2, chans, 3, padding=1), nn.ReLU(), nn.Conv2d(chans, chans, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(chans, chans, 3, padding=1), nn.ReLU(), nn.Conv2d(chans, 2, 3, padding=1))
        self.blocks = nn.ModuleList(block() for _ in range(cascades))

    def forward(self, kspace, mask):
        observed = mask * kspace
        x = ifft2c(observed)
        for block in self.blocks:
            update = block(torch.stack([x.real, x.imag], 1))
            x = x + torch.complex(update[:, 0], update[:, 1])
            x = ifft2c((1 - mask) * fft2c(x) + observed)
        return x


class Static(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(0.01 * torch.randn(SIZE, SIZE))

    def forward(self, kspace, labels):  # noqa: ARG002
        return self.logits.expand(len(kspace), -1, -1)


class Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(0.01 * torch.randn(2, SIZE, SIZE))

    def forward(self, kspace, labels):  # noqa: ARG002
        return self.logits[labels]


NET_INPUTS = ('raw', 'magnitude', 'label')
INPUT_WIDTH = {'raw': 2 * CENTRE_FEATURES ** 2, 'magnitude': CENTRE_FEATURES ** 2, 'label': 2}


class Net(nn.Module):
    """`raw`: real and imaginary parts of the acquired centre (the class needs a nonlinear feature: a linear probe is
    at chance); `magnitude`: their magnitudes (linearly decodable); `label`: the class one-hot (the table's input)."""

    def __init__(self, hidden=64, net_input='raw', init='zero'):
        super().__init__()
        low = SIZE // 2 - CENTRE_FEATURES // 2
        self.window = slice(low, low + CENTRE_FEATURES)
        self.net_input = net_input
        self.body = nn.Sequential(nn.Linear(INPUT_WIDTH[net_input], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, SIZE * SIZE)
        if init == 'zero':
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def features(self, kspace, labels=None):
        if self.net_input == 'label':
            return nn.functional.one_hot(labels, 2).float()
        centre = kspace[:, self.window, self.window].flatten(1)
        centre = centre / centre.abs().pow(2).sum(1, keepdim=True).sqrt().clamp(min=1e-8)
        if self.net_input == 'magnitude':
            return centre.abs()
        return torch.cat([centre.real, centre.imag], 1)

    def forward(self, kspace, labels):
        return self.head(self.body(self.features(kspace, labels))).view(-1, SIZE, SIZE)


SAMPLERS = {'static': Static, 'table': Table, 'net': Net}


def effective_probability(logits):
    available = (1 - centre_mask()).expand_as(logits)[:, None]
    probability = scores_to_prob(logits[:, None], FLOOR) * available
    return RescaleProbMap(probability, NEW / SIZE ** 2, available)[:, 0]


def draw(probability, threshold=None):
    return centre_mask() + threshold_random_mask(probability, FLOOR, threshold) * (1 - centre_mask())


def random_family(count, generator):
    """Broad random variable-density probability maps: row bands, column bands and isotropic profiles."""
    def log_uniform(low, high):
        return torch.exp(torch.empty(count, 1, 1).uniform_(torch.log(torch.tensor(low)).item(), torch.log(torch.tensor(high)).item(), generator=generator))
    offsets = (torch.arange(SIZE) - SIZE // 2).abs().float()
    ky, kx = offsets[None, :, None], offsets[None, None, :]
    power = torch.where(torch.rand(count, 1, 1, generator=generator) < 0.5, 1.0, 2.0)
    density = torch.exp(-(ky / log_uniform(0.7, 24.0)) ** power - (kx / log_uniform(0.7, 24.0)) ** power)
    density = density + 0.02 * torch.rand(count, 1, 1, generator=generator)
    available = (1 - centre_mask()).expand(count, SIZE, SIZE)[:, None]
    return RescaleProbMap(density[:, None] * available, NEW / SIZE ** 2, available)[:, 0]


def psnr(output, target):
    error = (output - target).abs().pow(2).flatten(1).mean(1)
    peak = target.abs().flatten(1).amax(1)
    return 10 * torch.log10(peak ** 2 / error)


def loss_of(recon, kspace, target, mask):
    return (recon(kspace, mask) - target).abs().pow(2).mean()


@torch.no_grad()
def evaluate(recon, sampler, val, swap):
    image, labels, kspace = val
    logits = sampler(kspace, labels)
    probability = effective_probability(logits)
    swapped = effective_probability(sampler(kspace[swap], labels[swap]))
    generator = torch.Generator().manual_seed(1234)
    own, other = torch.zeros(len(image)), torch.zeros(len(image))
    for _ in range(EVAL_DRAWS):
        threshold = torch.rand(probability.shape, generator=generator)
        own += psnr(recon(kspace, draw(probability, threshold)), image) / EVAL_DRAWS
        other += psnr(recon(kspace, draw(swapped, threshold)), image) / EVAL_DRAWS
    full = probability + centre_mask()
    means = [full[labels == c].mean(0) for c in (0, 1)]
    rows, cols = axis_bands()
    own_band = [float((means[0] * rows).sum() / (means[0] * (1 - centre_mask())).sum()),
                float((means[1] * cols).sum() / (means[1] * (1 - centre_mask())).sum())]
    centred = [m - m.mean() for m in means]
    correlation = float((centred[0] * centred[1]).sum() / (centred[0].norm() * centred[1].norm()))
    per_slice = full - torch.stack(means)[labels]
    return {'psnr': float(own.mean()), 'psnr_by_class': [float(own[labels == c].mean()) for c in (0, 1)],
            'class_mask_corr': correlation, 'own_band_share': own_band,
            'swap_gain_db': float((own - other).mean()),
            'within_class_std': float(per_slice.std(0).mean())}


def validation_set(purity):
    image, labels = images(VAL_IMAGES, purity, torch.Generator().manual_seed(777))
    return image, labels, fft2c(image.to(torch.complex64))


def swap_index(labels):
    generator = torch.Generator().manual_seed(4321)
    swap = torch.zeros_like(labels)
    for c in (0, 1):
        pool = torch.nonzero(labels != c).flatten()
        rows = torch.nonzero(labels == c).flatten()
        swap[rows] = pool[torch.randint(len(pool), (len(rows),), generator=generator)]
    return swap


def load_sampler_init(sampler, path):
    """Start `sampler` at the masks of a saved static or table sampler: per-class maps for table, the output bias for net."""
    saved = torch.load(path / 'sampler.pt')['logits']
    maps = saved.expand(2, SIZE, SIZE) if saved.dim() == 2 else saved
    with torch.no_grad():
        if isinstance(sampler, Net):
            sampler.head.bias.copy_(maps.mean(0).flatten())
        elif isinstance(sampler, Table):
            sampler.logits.copy_(maps)
        else:
            sampler.logits.copy_(maps.mean(0))


def probe(args):
    """Held-out accuracy of decoding the class from each net input: logistic regression and a 64-unit MLP."""
    def inputs(count, seed):
        image, labels = images(count, args.purity, torch.Generator().manual_seed(seed))
        return fft2c(image.to(torch.complex64)), labels

    (k_train, y_train), (k_test, y_test) = inputs(8000, 1), inputs(2000, 2)
    results = {}
    for net_input in ('raw', 'magnitude'):
        net = Net(net_input=net_input)
        train, test = net.features(k_train), net.features(k_test)
        mean, std = train.mean(0), train.std(0) + 1e-6
        train, test = (train - mean) / std, (test - mean) / std
        accuracy = {}
        for name, model in (('linear', nn.Linear(train.shape[1], 1)),
                            ('mlp', nn.Sequential(nn.Linear(train.shape[1], 64), nn.ReLU(), nn.Linear(64, 1)))):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(1000):
                optimizer.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(model(train)[:, 0], y_train.float()).backward()
                optimizer.step()
            with torch.no_grad():
                accuracy[name] = float(((model(test)[:, 0] > 0).long() == y_test).float().mean())
        results[net_input] = accuracy
        print(net_input, accuracy, flush=True)
    (args.out / 'probe.json').write_text(json.dumps({'purity': args.purity, 'accuracy': results}, indent=1))


def regress(args):
    """Can the net represent the per-class maps a table learned? Fit it to them by direct regression on the logits."""
    target_maps = torch.load(args.sampler_init / 'sampler.pt')['logits']
    image, labels = images(4096, args.purity, torch.Generator().manual_seed(args.seed))
    kspace = fft2c(image.to(torch.complex64))
    results = {}
    for net_input in NET_INPUTS:
        net = Net(net_input=net_input, init=args.net_init)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(args.steps):
            rows = torch.randint(len(labels), (args.batch,))
            optimizer.zero_grad()
            (net(kspace[rows], labels[rows]) - target_maps[labels[rows]]).pow(2).mean().backward()
            optimizer.step()
        with torch.no_grad():
            fitted = net(kspace, labels)
            target = target_maps[labels]
            class_difference = (target_maps[0] - target_maps[1]).norm()
            fitted_difference = (fitted[labels == 0].mean(0) - fitted[labels == 1].mean(0)).norm()
            results[net_input] = {'relative_error': float((fitted - target).norm() / target.norm()),
                                  'class_difference_fitted_over_target': float(fitted_difference / class_difference)}
        print(net_input, results[net_input], flush=True)
    (args.out / 'regress.json').write_text(json.dumps({'target': str(args.sampler_init), 'steps': args.steps, 'results': results}, indent=1))


def gradients(args):
    """At a saved static mask, split each image's gradient on the score map into population, class offset,
    image-specific and draw-noise parts (energies from independent half-draw means, as in mask_gradient_anatomy_r8.py),
    through the reconstructor of `--recon-init`. Also the alignment of the class offset with the ideal adaptive move
    (toward the own axis band, away from the other)."""
    recon = Recon()
    recon.load_state_dict(torch.load(args.recon_init / 'recon.pt'))
    recon.requires_grad_(False)
    logits = torch.load(args.sampler_init / 'sampler.pt')['logits']
    logits = logits.mean(0) if logits.dim() == 3 else logits
    image, labels = images(256, args.purity, torch.Generator().manual_seed(args.seed + 11))
    kspace = fft2c(image.to(torch.complex64))
    halves = torch.zeros(2, len(image), SIZE, SIZE)
    squares = torch.zeros(len(image), SIZE, SIZE)
    generator = torch.Generator().manual_seed(args.seed)
    for draw_index in range(args.draws):
        z = logits.expand(len(image), -1, -1).clone().requires_grad_(True)
        threshold = torch.rand(len(image), 1, SIZE, SIZE, generator=generator)
        mask = draw(effective_probability(z), threshold[:, 0])
        per_image = (recon(kspace, mask) - image).abs().pow(2).flatten(1).mean(1)
        per_image.sum().backward()
        halves[draw_index % 2] += z.grad / (args.draws / 2)
        squares += z.grad ** 2
    mean = halves.mean(0)
    noise_per_draw = float(((squares / args.draws - mean ** 2) * args.draws / (args.draws - 1)).flatten(1).sum(1).mean())

    def parts(g):
        population = g.mean(0, keepdim=True)
        offset = torch.zeros_like(g)
        for c in (0, 1):
            offset[labels == c] = g[labels == c].mean(0, keepdim=True) - population
        return {'pop': population.expand_as(g), 'class': offset, 'image': g - population - offset}

    a, b = parts(halves[0]), parts(halves[1])
    energy = {k: float((a[k] * b[k]).flatten(1).sum(1).mean()) for k in a}
    total = sum(energy.values())
    rows, cols = axis_bands()
    ideal = torch.where((labels == 0)[:, None, None], rows - cols, cols - rows)  # descent direction: raise own band
    offset = -(a['class'] + b['class']) / 2
    alignment = float(((offset * ideal).flatten(1).sum(1) / (offset.flatten(1).norm(dim=1) * ideal.flatten(1).norm(dim=1))).mean())
    results = {'recon': str(args.recon_init), 'mask': str(args.sampler_init), 'draws': args.draws,
               'energy': energy, 'fractions': {k: v / total for k, v in energy.items()},
               'noise_per_draw': noise_per_draw, 'class_snr_per_draw': energy['class'] / noise_per_draw,
               'class_offset_alignment_with_ideal': alignment}
    print(json.dumps(results), flush=True)
    (args.out / 'gradients.json').write_text(json.dumps(results, indent=1))


def main():
    args = parse_args()
    if args.stage == 'gradients':
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        gradients(args)
        return
    if args.stage == 'regress':
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        regress(args)
        return
    if args.stage == 'probe':
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        probe(args)
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    recon = Recon()
    if args.recon_init is not None:
        recon.load_state_dict(torch.load(args.recon_init / 'recon.pt'))
    recon.requires_grad_(not args.freeze_recon)
    val = validation_set(args.purity)
    swap = swap_index(val[1])
    history = []
    started = time.time()

    if args.stage == 'generalist':
        optimizer = torch.optim.Adam(recon.parameters(), lr=args.lr)
        for step in range(args.steps + 1):
            if step % args.eval_every == 0:
                with torch.no_grad():
                    probability = random_family(VAL_IMAGES, torch.Generator().manual_seed(99))
                    score = psnr(recon(val[2], draw(probability, torch.rand(probability.shape, generator=torch.Generator().manual_seed(5)))), val[0])
                history.append({'step': step, 'psnr_random_family': float(score.mean())})
                print(json.dumps(history[-1]), flush=True)
            if step == args.steps:
                break
            image, _ = images(args.batch, args.purity, generator)
            mask = draw(random_family(args.batch, generator)).detach()
            optimizer.zero_grad()
            loss_of(recon, fft2c(image.to(torch.complex64)), image, mask).backward()
            optimizer.step()
        torch.save(recon.state_dict(), args.out / 'recon.pt')
        (args.out / 'history.json').write_text(json.dumps({'config': vars(args) | {'out': str(args.out)}, 'history': history}, indent=1, default=str))
        return

    sampler = Net(net_input=args.net_input, init=args.net_init) if args.sampler == 'net' else SAMPLERS[args.sampler]()
    if args.sampler_init is not None:
        load_sampler_init(sampler, args.sampler_init)
    groups = [{'params': [p for n, p in sampler.named_parameters() if n in ('logits', 'head.bias')], 'lr': args.logit_lr},
              {'params': [p for n, p in sampler.named_parameters() if n not in ('logits', 'head.bias')], 'lr': args.lr}]
    if not args.freeze_recon:
        groups.append({'params': list(recon.parameters()), 'lr': args.lr})
    optimizer = torch.optim.Adam([g for g in groups if g['params']])
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            history.append({'step': step, **evaluate(recon, sampler, val, swap)})
            print(json.dumps(history[-1]), flush=True)
        if step == args.steps:
            break
        image, labels = images(args.batch, args.purity, generator)
        kspace = fft2c(image.to(torch.complex64))
        mask = draw(effective_probability(sampler(kspace, labels)))
        optimizer.zero_grad()
        loss_of(recon, kspace, image, mask).backward()
        optimizer.step()

    torch.save(recon.state_dict(), args.out / 'recon.pt')
    if not isinstance(sampler, Net):
        torch.save({'logits': sampler.logits.detach()}, args.out / 'sampler.pt')
    torch.save(sampler.state_dict(), args.out / 'sampler_state.pt')
    summary = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'seconds': time.time() - started, 'final': history[-1], 'history': history}
    (args.out / 'history.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
