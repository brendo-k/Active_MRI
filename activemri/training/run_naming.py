"""wandb run names, groups and tags for Sequential_Trainer runs.

A name leads with what most often tells runs apart (sampler, acceleration) and then lists only the
settings that differ from BASELINE or the config defaults, so runs in the same family read alike and
their differences are visible before the wandb sidebar truncates them:

    conv2d-1step_R8_head-bias_floor0.05_brain+knee
    loupe2d_R8_slope5_dc-multiply_train-brain_lr0.0003_ep30_brain+knee

Settings are read with getattr defaults, so options from older checkpoints and configs stored in
wandb before a field existed still name correctly.
"""
from activemri.config import TrainConfig

BASELINE = {
    'recon_type': 'varnet',
    'lr': 1e-3,
    'batch_size': 16,
    'num_epochs': 50,
    'lr_scheduler': 'step',
    'resolution': [256, 256],
    'seed': 42,
}
NON_ADAPTIVE_SAMPLERS = ('loupe_2d', 'loupe_1d', 'autosamp_2d', 'oracle')
IMAGE_INPUT_SAMPLERS = ('conv_2d', 'mnet_2d')
VARNET_DEFAULTS = {'num_cascades': 12, 'chans': 18, 'pools': 4, 'dc_mode': 'where'}
_DEFAULTS = TrainConfig()


def _get(options, key):
    return getattr(options, key, getattr(_DEFAULTS, key, None))


def _recon(options):
    recon = _get(options, 'recon')
    return recon if isinstance(recon, dict) else vars(recon)


def _number(value):
    return f'{int(value)}' if float(value) == int(value) else f'{value:g}'


def _differs(options, key):
    return _get(options, key) != getattr(_DEFAULTS, key)


def sampler_label(options) -> str:
    sampler = _get(options, 'sampler_type')
    if _get(options, 'ppo_policy'):
        return 'ppo-policy'
    if _get(options, 'mask_train_mode') == 'random':
        return 'random-mask'
    label = sampler.replace('_', '')
    if _get(options, 'mask_train_mode') == 'reinforce':
        label += '-reinforce'
    if sampler not in NON_ADAPTIVE_SAMPLERS and not _get(options, 'fixed_input'):
        label += f"-{_get(options, 'num_step')}step"
    return label


def _sampler_tokens(options):
    sampler = _get(options, 'sampler_type')
    tokens = []
    if sampler in IMAGE_INPUT_SAMPLERS and not _get(options, 'sampler_image_input'):
        tokens.append('kspace-in')
    if _get(options, 'sampler_head') != 'none':
        tokens.append(f"head-{_get(options, 'sampler_head')}")
    if (_get(options, 'sampler_chans'), _get(options, 'sampler_depth')) != (64, 4):
        tokens.append(f"samp-c{_get(options, 'sampler_chans')}d{_get(options, 'sampler_depth')}")
    if _get(options, 'anatomy_conditioning'):
        tokens.append('cond-anat')
    if _get(options, 'fixed_mask_checkpoints'):
        tokens.append('fixed-masks')
    if sampler.startswith('loupe') and _differs(options, 'slope'):
        tokens.append(f"slope{_get(options, 'slope'):g}")
    mask_floor, ste_floor = _get(options, 'mask_grad_floor'), _get(options, 'ste_grad_floor')
    if mask_floor == ste_floor and mask_floor > 0:
        tokens.append(f'floor{mask_floor:g}')
    else:
        tokens += [f'{name}{floor:g}' for name, floor in (('mfloor', mask_floor), ('stefloor', ste_floor)) if floor > 0]
    if _get(options, 'mask_draws') > 1:
        tokens.append(f"draws{_get(options, 'mask_draws')}")
    if _differs(options, 'logit_limit') or _differs(options, 'logit_leak'):
        tokens.append(f"clamp{_get(options, 'logit_limit'):g}-{_get(options, 'logit_leak'):g}")
    if _get(options, 'caipi_keep_fraction') != 1.0:
        tokens.append(f"pf{_get(options, 'caipi_keep_fraction'):g}")
    return tokens + _family_tokens(options, sampler)


def _family_tokens(options, sampler):
    """Settings that only one sampler family reads, named only for that family."""
    if _get(options, 'mask_train_mode') == 'reinforce':
        fields = [('reinforce_entropy_weight', 'ent'), ('reinforce_group_size', 'group'),
                  ('reinforce_vd_power', 'vd'), ('reinforce_lr', 'plr'), ('reinforce_eval', 'eval-')]
        tokens = ['advnorm'] if _get(options, 'reinforce_normalize_advantages') else []
    elif sampler == 'autosamp_2d':
        fields = [('autosamp_init', 'init-'), ('autosamp_lr', 'plr'), ('autosamp_kernel_sigma_start', 'anneal')]
        tokens = []
    elif sampler == 'mnet_2d':
        fields = [('mnet_accept', 'accept-'), ('mnet_label_steps', 'T'), ('mnet_sampler_steps', 'c'),
                  ('mnet_init_margin', 'margin'), ('mnet_sparsity_weight', 'sp'), ('mnet_consistency_weight', 'cons'),
                  ('mnet_label_lr', 'labellr'), ('mnet_label_binarize', 'bin-')]
        tokens = [name for key, name in (('mnet_compare_random', 'no-random-check'), ('mnet_rollback', 'no-rollback'),
                                         ('mnet_degeneracy_fallback', 'no-fallback')) if not _get(options, key)]
        tokens += ['random-retry'] if _get(options, 'mnet_random_retry') else []
    else:
        return []
    for key, name in fields:
        value = _get(options, key)
        if _differs(options, key):
            tokens.append(f'{name}{value:g}' if isinstance(value, (int, float)) else f'{name}{value}')
    return tokens


def _recon_tokens(options):
    recon = _recon(options)
    recon_type = recon.get('recon_type')
    tokens = [] if recon_type == BASELINE['recon_type'] else [recon_type]
    if recon_type == 'varnet':
        tokens += [f'dc-{recon[key]}' if key == 'dc_mode' else f'{key}{recon[key]}'
                   for key, default in VARNET_DEFAULTS.items() if recon.get(key, default) != default]
        tokens += [name for key, name in (('anatomy_conditioning', 'recon-cond'), ('film_conditioning', 'film'),
                                          ('adaptive_input', 'aha-in'), ('dc_weight_map', 'dc-map')) if recon.get(key)]
    if _get(options, 'freeze_recon'):
        tokens.append('frozen-recon')
    if _get(options, 'pretrained_recon'):
        tokens.append('pretrained-recon')
    return tokens


def _loss_tokens(options):
    tokens = [] if _get(options, 'loss_type') == 'l1' else [f"loss-{_get(options, 'loss_type')}"]
    if _get(options, 'uncertainty_loss'):
        tokens.append('unc')
    if _get(options, 'anatomy_weight') > 0:
        tokens.append(f"anat{_get(options, 'anatomy_weight'):g}")
    if _get(options, 'ssdu'):
        tokens.append('pseudo-target' if _get(options, 'pseudo_target') else 'ssdu')
    if _get(options, 'phase_scramble_std') > 0:
        tokens.append(f"phase{_get(options, 'phase_scramble_std'):g}")
    if _get(options, 'random_rotate'):
        tokens.append('rotate')
    return tokens


def _training_tokens(options):
    tokens = []
    train_anatomies = _get(options, 'train_anatomies')
    if train_anatomies:
        tokens.append('train-' + '+'.join(train_anatomies))
    if _get(options, 'lr') != BASELINE['lr']:
        tokens.append(f"lr{_get(options, 'lr'):g}")
    if _get(options, 'lr_scheduler') != BASELINE['lr_scheduler']:
        tokens.append(_get(options, 'lr_scheduler').replace('_', '-'))
    if _get(options, 'weight_decay'):
        tokens.append(f"wd{_get(options, 'weight_decay'):g}")
    if _get(options, 'num_epochs') != BASELINE['num_epochs']:
        tokens.append(f"ep{_get(options, 'num_epochs')}")
    if _get(options, 'batch_size') != BASELINE['batch_size']:
        tokens.append(f"bs{_get(options, 'batch_size')}")
    if list(_get(options, 'resolution')) != BASELINE['resolution']:
        tokens.append('x'.join(str(size) for size in _get(options, 'resolution')))
    if _get(options, 'seed') not in (None, BASELINE['seed']):
        tokens.append(f"seed{_get(options, 'seed')}")
    return tokens


def _acceleration(options):
    return 'R' + 'x'.join(_number(a) for a in _get(options, 'accelerations'))


def _data(options):
    return '+'.join(_get(options, 'condition_classes') or _get(options, 'anatomy_classes'))


def run_name(options) -> str:
    if _get(options, 'run_name'):
        return _get(options, 'run_name')
    tokens = [sampler_label(options), _acceleration(options), *_sampler_tokens(options), *_recon_tokens(options),
              *_loss_tokens(options), *_training_tokens(options), _data(options)]
    if _get(options, 'checkpoint'):
        tokens.append('resumed')
    return '_'.join(token for token in tokens if token)


def run_group(options):
    """The directory holding the run's out_dir, e.g. `grad_floor_compare`, so sweeps sit together; None at top level."""
    out_dir = _get(options, 'out_dir')
    parent = str(out_dir[0] if isinstance(out_dir, (list, tuple)) else out_dir).rstrip('/').rsplit('/', 2)
    return parent[-2] if len(parent) > 1 and parent[-2] not in ('checkpoints', '.', '') else None


def run_tags(options):
    return [sampler_label(options).split('-')[0], _recon(options).get('recon_type'), _acceleration(options),
            *(_get(options, 'anatomy_classes') or [])]
