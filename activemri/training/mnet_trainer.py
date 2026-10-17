"""Alternating sampler / reconstructor training of Huang & Ravishankar, "Single-Pass Object-Adaptive
Data Undersampling and Reconstruction for MRI" (IEEE TCI 2022), Algorithms 1 and 2.

The sampler (MNet, `sampler_type=mnet_2d`) is never trained through the reconstruction loss. Each batch:

  1. MNet proposes a mask M_s (exact budget, no grad).
  2. mask-backward: a free logit map xi per slice, started at M_s, is optimised for `mnet_label_steps`
     steps against the reconstruction loss (straight-through through the budgeted top-k), together with
     the reconstructor. The refined mask M_ref is the label.
  3. The label is accepted if its quality Q = -NRMSE beats M_s's (and a random mask's, scored by the
     frozen warm-start reconstructor). Accepted: MNet takes `mnet_sampler_steps` BCE steps towards the
     label and the reconstructor keeps its update. Rejected: the reconstructor update is rolled back.

Every learning rate, optimizer and schedule is the project's (Adam at `training.lr`, the same clip).
"""
import copy

import torch
import torch.nn.functional as F

from activemri.training.trainer import GRAD_CLIP_NORM, Sequential_Trainer, build_model, zero_coil_fraction
from activemri.utils.mask_binarization import topk_straight_through

ACCEPT_MODES = ('q_check', 'always')
LABEL_BINARIZATIONS = ('threshold', 'topk')
BCE_EPS = 1e-6


def normalized_rmse(output, target):
    """Per-example ||output - target|| / ||target||, [B]."""
    if output.dim() != target.dim():
        target = target.unsqueeze(1)
    return (output - target).flatten(1).norm(dim=1) / target.flatten(1).norm(dim=1).clamp(min=1e-12)


def accept_label(q_refined, q_mnet, q_random, mode, compare_random):
    """Algorithm 2, line 7: whether the refined mask may train MNet. Qs are per-example, higher is better."""
    if mode == 'always':
        return True
    best_other = q_mnet.mean()
    if compare_random:
        best_other = torch.maximum(best_other, q_random.mean())
    return bool(q_refined.mean() > best_other)


def distinct_rows(masks):
    """Number of distinct masks in a [B, ...] batch and the size of the most repeated one."""
    _, counts = torch.unique(masks.flatten(1), dim=0, return_counts=True)
    return len(counts), int(counts.max())


def is_degenerate(masks):
    """Remark 5: more than half of the batch's masks are the same."""
    return distinct_rows(masks)[1] > masks.shape[0] / 2


def random_new_locations(available, count):
    """[B, 1, H, W] binary mask of `count` uniformly random locations among the available ones."""
    return topk_straight_through(torch.rand_like(available), count, available).detach()


def masked_bce_mean(probabilities, targets, available):
    """BCE(probabilities, soft targets) averaged over the available locations of each example, then over the batch."""
    probabilities = probabilities.clamp(BCE_EPS, 1 - BCE_EPS)
    bce = -(targets * probabilities.log() + (1 - targets) * (1 - probabilities).log())
    return ((bce * available).flatten(1).sum(dim=1) / available.flatten(1).sum(dim=1)).mean()


def label_mask(label_probabilities, old_mask, binarize, exact_budget_sampler=None, budget=None, valid_mask=None):
    """Mask the reconstructor sees for label probabilities in (0, 1): their 0.5 threshold, or the budgeted top-k."""
    if binarize == 'topk':
        return exact_budget_sampler.select_locations(label_probabilities, old_mask, budget, valid_mask)[1]
    hard = (label_probabilities > 0.5).to(label_probabilities.dtype)
    return torch.clamp(old_mask + hard + (label_probabilities - label_probabilities.detach()), min=0, max=1)


class MNetTrainer(Sequential_Trainer):
    def __init__(self, options, train_loader, device, val_loader=None):
        if options.sampler_type != 'mnet_2d' or options.num_step != 1:
            raise ValueError("MNetTrainer trains sampler_type=mnet_2d with num_step=1")
        if options.mnet_accept not in ACCEPT_MODES:
            raise ValueError(f"Unknown mnet_accept {options.mnet_accept!r}; expected one of {ACCEPT_MODES}")
        if options.mnet_label_binarize not in LABEL_BINARIZATIONS:
            raise ValueError(f"Unknown mnet_label_binarize {options.mnet_label_binarize!r}; expected one of {LABEL_BINARIZATIONS}")
        if options.data_parallel or options.ssdu or options.pseudo_target or options.freeze_recon:
            raise ValueError("MNetTrainer does not support data_parallel, ssdu, pseudo_target or freeze_recon")
        self.compares_random = options.mnet_accept == 'q_check' and options.mnet_compare_random
        if self.compares_random and options.pretrained_recon is None:
            raise ValueError(
                "mnet_compare_random scores a random mask with the frozen warm-start reconstructor, so "
                "output.pretrained_recon must be a reconstructor trained on random masks "
                "(or set sampler.mnet_compare_random=false)")
        super().__init__(options, train_loader, device, val_loader=val_loader)

        self.recon_modules = self._recon_modules()
        self.recon_params = [p for module in self.recon_modules for p in module.parameters() if p.requires_grad]
        self.sampler_params = [p for p in self.model.sampler.parameters() if p.requires_grad]
        self.budget_count = int(round(self.model.budget))
        self.warm_start = None
        if self.compares_random:
            self.warm_start = build_model(options).to(device).requires_grad_(False).eval()
            recon_state = {k: v for k, v in self.model.state_dict().items() if not k.startswith('sampler.')}
            self.warm_start.load_state_dict(recon_state, strict=False)
        self.batches_seen = 0
        self.batches_accepted = 0

    # ---- the per-batch alternating step -------------------------------------------------

    def _train_forward(self, target, kspace, batch):
        """The mnet mask's reconstruction before the update: what the step's loss and metrics describe."""
        with torch.no_grad():
            return self.model(target, kspace, **self._model_inputs(batch))

    def _optimize(self, loss, batch, kspace, target, step):  # noqa: ARG002
        options = self.options
        inputs = self._model_inputs(batch)
        labels = inputs['anatomy_labels']
        target_dict = self._target_dict(batch, target, kspace)
        meta = {'recon_weight': 1.0, 'kspace_weight': 0.0, 'uncertainty_weight': 0.0, 'anatomy_weight': 0.0}
        stats = {}

        with torch.no_grad():
            proposal = self.model.propose_mask(kspace, labels)
        old_mask, valid_mask = proposal['old_mask'], proposal['valid_mask']
        available = (1 - old_mask) * valid_mask
        mnet_mask = proposal['mask'].detach()
        mnet_new = mnet_mask - old_mask
        mnet_prob = proposal['rescaled'].detach()

        distinct, most_repeated = distinct_rows(mnet_new)
        stats['mnet/distinct_masks_frac'] = distinct / mnet_new.shape[0]
        fallback = options.mnet_degeneracy_fallback and most_repeated > mnet_new.shape[0] / 2
        stats['mnet/degenerate_fallback'] = float(fallback)
        initial_new = random_new_locations(available, self.budget_count) if fallback else mnet_new

        snapshot = self._snapshot()
        quality = lambda model, mask: -self._nrmse(model, mask, target, kspace, inputs)
        refine = lambda initial: self._mask_backward(
            initial, mnet_prob, old_mask, valid_mask, available, target, kspace, inputs, target_dict, meta)

        refined_mask, stats_labels = refine(initial_new)
        with torch.no_grad():
            q_mnet = quality(self.model, mnet_mask)
            q_refined = quality(self.model, refined_mask)
        retried = (options.mnet_random_retry and not fallback and bool(q_refined.mean() <= q_mnet.mean()))
        if retried:
            self._restore(snapshot)
            refined_mask, stats_labels = refine(random_new_locations(available, self.budget_count))
            with torch.no_grad():
                q_mnet = quality(self.model, mnet_mask)
                q_refined = quality(self.model, refined_mask)
        stats.update(stats_labels)
        stats['mnet/random_retry'] = float(retried)
        stats['mnet/flip_frac'] = float(((refined_mask - old_mask) * (1 - mnet_new)).sum() / mnet_new.sum().clamp(min=1))

        q_random = None
        if self.compares_random:
            with torch.no_grad():
                random_mask = old_mask + random_new_locations(available, self.budget_count)
                q_random = quality(self.warm_start, random_mask)
        accepted = accept_label(q_refined, q_mnet, q_random, options.mnet_accept, self.compares_random)
        stats.update({'mnet/accepted': float(accepted), 'mnet/q_refined': q_refined.mean().item(),
                      'mnet/q_mnet': q_mnet.mean().item()})
        if q_random is not None:
            stats['mnet/q_random'] = q_random.mean().item()

        if accepted:
            stats['mnet/sampler_bce'] = self._fit_sampler(refined_mask - old_mask, old_mask, available, kspace, labels)
        elif options.mnet_rollback:
            self._restore(snapshot)

        self.batches_seen += 1
        self.batches_accepted += accepted
        stats['mnet/accept_rate_running'] = self.batches_accepted / self.batches_seen
        stats['data/zero_coil_frac'] = zero_coil_fraction(kspace)
        stats['train/lr'] = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            f"step {step}: {'accept' if accepted else 'reject'} (running {stats['mnet/accept_rate_running']:.2f}) "
            f"Q refined/mnet/random = {stats['mnet/q_refined']:.4f}/{stats['mnet/q_mnet']:.4f}/{stats.get('mnet/q_random', float('nan')):.4f} "
            f"flip={stats['mnet/flip_frac']:.3f} distinct={stats['mnet/distinct_masks_frac']:.2f} "
            f"recon loss {stats['mnet/recon_loss_first']:.4f}->{stats['mnet/recon_loss_last']:.4f} "
            f"loop new locations {stats['mnet/loop_new_locations_last']:.1f}/{self.budget_count}")
        return stats

    def _mask_backward(self, initial_new, mnet_prob, old_mask, valid_mask, available, target, kspace, inputs,
                       target_dict, meta):
        """Algorithm 1: optimise per-slice label logits (and the reconstructor) from the mnet mask; returns the refined mask."""
        options = self.options
        sampler = self.model.sampler
        xi = (options.mnet_init_margin * (2 * initial_new - 1)).requires_grad_(True)
        xi_start = xi.detach().clone()
        xi_optimizer = torch.optim.Adam([xi], lr=options.lr if options.mnet_label_lr is None else options.mnet_label_lr)
        stats = {'mnet/skipped_nonfinite_steps': 0.0}

        for iteration in range(options.mnet_label_steps):
            label_prob = torch.sigmoid(xi) * available
            mask = label_mask(label_prob, old_mask, options.mnet_label_binarize, sampler, self.model.budget, valid_mask)
            pred_dict = self.model(target, kspace, fixed_mask=mask, **inputs)
            recon_loss, _ = self.model.loss(pred_dict, target_dict, meta, options.loss_type)
            sparsity = label_prob.flatten(1).sum(dim=1).mean()
            consistency = masked_bce_mean(torch.sigmoid(xi), mnet_prob, available)
            total = recon_loss + options.mnet_sparsity_weight * sparsity + options.mnet_consistency_weight * consistency
            if not self._step(total, self.recon_params, xi_optimizer):
                stats['mnet/skipped_nonfinite_steps'] += 1
            if iteration == 0:
                stats['mnet/recon_loss_first'] = recon_loss.item()
            loop_new_locations = ((mask.detach() - old_mask) * valid_mask).flatten(1).sum(dim=1).mean().item()
        stats['mnet/recon_loss_last'] = recon_loss.item()
        stats['mnet/loop_new_locations_last'] = loop_new_locations
        stats['mnet/label_logit_shift'] = ((xi.detach() - xi_start).abs() * available).sum().item() / available.sum().item()

        with torch.no_grad():
            _, refined_mask = sampler.select_locations(torch.sigmoid(xi) * available, old_mask, self.model.budget, valid_mask)
        return refined_mask.detach(), stats

    def _fit_sampler(self, label, old_mask, available, kspace, labels):
        """`mnet_sampler_steps` BCE steps of MNet's budgeted probabilities towards the binary label; the last loss."""
        for _ in range(self.options.mnet_sampler_steps):
            prob = self.model.propose_mask(kspace, labels)['rescaled']
            loss = F.binary_cross_entropy(prob.clamp(0, 1), label, weight=available, reduction='sum') / available.sum()
            self._step(loss, self.sampler_params)
        return loss.item()

    def _step(self, loss, clip_params, *extra_optimizers):
        """Backward, clip and step (the project's optimizer plus any extra); False if the gradient was not finite."""
        optimizers = (self.optimizer, *extra_optimizers)
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(clip_params, GRAD_CLIP_NORM)
        finite = bool(torch.isfinite(norm))
        if finite:
            for optimizer in optimizers:
                optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad()
        return finite

    @staticmethod
    def _nrmse(model, mask, target, kspace, inputs):
        return normalized_rmse(model(target, kspace, fixed_mask=mask, **inputs)['output'], target)

    # ---- reconstructor rollback ---------------------------------------------------------

    def _snapshot(self):
        """Reconstructor and sensitivity-net weights and their Adam state, to restore if the label is rejected."""
        weights = [copy.deepcopy(module.state_dict()) for module in self.recon_modules]
        state = {p: copy.deepcopy(self.optimizer.state[p]) for p in self.recon_params if p in self.optimizer.state}
        return weights, state

    def _restore(self, snapshot):
        weights, state = snapshot
        for module, saved in zip(self.recon_modules, weights):
            module.load_state_dict(saved)
        for p in self.recon_params:
            if p in state:
                self.optimizer.state[p] = state[p]
            else:
                self.optimizer.state.pop(p, None)
