"""Can run-state code resume a saved checkpoint's optimizer exactly? Builds each run's model from its saved options,
forms the trainer's parameter groups and optimizer, and compares every group's parameter count and each parameter's
shape, in order, with the saved optimizer state's (Adam's exp_avg, Muon's momentum buffer).
`optimizer.load_state_dict` raises on a count mismatch but silently misassigns state between same-shaped parameters, so
a resume needs this to hold exactly. It then loads the saved state as the trainer's resume does, and checks that every
parameter's restored state equals the saved one, key for key, and that every group keeps its saved lr.

    python scripts/diagnostics/resume_param_groups_check.py /export/scratch/bkadota/multistep_adaptivity_sync/runs/F_4_s42_fscr1_tF ...
"""
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from activemri.training.trainer import Sequential_Trainer, build_model, build_optimizer  # noqa: E402


def same(value, saved):
    return torch.equal(value, saved) if torch.is_tensor(value) and torch.is_tensor(saved) else value == saved


def restored_problems(optimizer, saved):
    expected = copy.deepcopy(saved)
    try:
        optimizer.load_state_dict(saved)
    except Exception as error:  # the trainer's resume would stop here
        return [f'load_state_dict raised {type(error).__name__}: {error}']
    problems = []
    for index, (group, saved_group) in enumerate(zip(optimizer.param_groups, expected['param_groups'])):
        if group['lr'] != saved_group['lr']:
            problems.append(f"group {index}: lr {group['lr']} against {saved_group['lr']} saved")
        for position, (parameter, key) in enumerate(zip(group['params'], saved_group['params'])):
            state, saved_state = optimizer.state.get(parameter, {}), expected['state'].get(key, {})
            if state.keys() != saved_state.keys() or not all(same(state[name], saved_state[name]) for name in state):
                problems.append(f'group {index} position {position}: restored state {sorted(state)} differs from the '
                                f'saved {sorted(saved_state)}')
    return problems


def check(run):
    checkpoint = torch.load(Path(run) / 'last_model.pt', map_location='cpu', weights_only=False)
    options = checkpoint['options']
    trainer = Sequential_Trainer.__new__(Sequential_Trainer)
    trainer.model, trainer.options = build_model(options), options
    trainer.model.load_state_dict(checkpoint['model'])
    optimizer = build_optimizer(options, trainer._param_groups(), trainer.model)
    saved = checkpoint['optimizer']
    problems = []
    if len(saved['param_groups']) != len(optimizer.param_groups):
        problems.append(f"{len(optimizer.param_groups)} groups against {len(saved['param_groups'])} saved")
    for index, (group, saved_group) in enumerate(zip(optimizer.param_groups, saved['param_groups'])):
        if len(group['params']) != len(saved_group['params']):
            problems.append(f"group {index}: {len(group['params'])} parameters against {len(saved_group['params'])}")
            continue
        for position, (parameter, key) in enumerate(zip(group['params'], saved_group['params'])):
            state = saved['state'].get(key, {})
            buffer = next((state[name] for name in ('exp_avg', 'momentum_buffer') if name in state), None)
            if buffer is None or tuple(buffer.shape) != tuple(parameter.shape):
                problems.append(f'group {index} position {position}: {tuple(parameter.shape)} against '
                                f'{None if buffer is None else tuple(buffer.shape)}')
    if not problems:
        problems = restored_problems(optimizer, saved)
    sizes = [len(group['params']) for group in optimizer.param_groups]
    lrs = [group['lr'] for group in optimizer.param_groups]
    print(f"{Path(run).name}: groups {sizes}, lr {lrs}; " + ('; '.join(problems[:5]) if problems else
                                                            'every parameter matches and resumes its saved state'))
    return not problems


if __name__ == '__main__':
    sys.exit(0 if all([check(run) for run in sys.argv[1:]]) else 1)
