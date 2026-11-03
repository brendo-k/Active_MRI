"""Queue shim for the varnet_train_speed worker: run the paired eval of every finished run on the worker's GPU.

The worker runs `train.py --model sequential <overrides> output.out_dir=<dir> ...`; a copy of this file named train.py
in a directory listed as the job's code snapshot does instead:
  1. waits until each run in `wait_for=a,b,...` has finished (its log reached its last epoch) or given up, or until
     `until=HH:MM`; `wait_for=all` waits for every other claimed job of the queue;
  2. runs scripts/eval/varnet_train_speed_eval.py from `snapshot=<dir>` once per control arm (control, control_short,
     control_10ep, full_control), all sharing one per-slice cache in the queue directory, writing into output.out_dir;
  3. prints the last-epoch line the worker checks.
"""
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path

args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
out = Path(args['output.out_dir'])
queue = out.parent
snapshot = args['snapshot']
out.mkdir(parents=True, exist_ok=True)
until = datetime.datetime.combine(datetime.date.today(), datetime.time.fromisoformat(args.get('until', '06:25')))
if until < datetime.datetime.now() - datetime.timedelta(hours=12):
    until += datetime.timedelta(days=1)


def settled(name):
    claim = queue / 'claims' / name
    if (claim / 'done').exists() or (claim / 'gave_up').exists():
        return True
    log = queue / f'{name}.log'
    epochs = re.findall(r'Epoch = \[ *(\d+)/ *(\d+)\]', log.read_text()) if log.exists() else []
    return bool(epochs) and epochs[-1][0] == epochs[-1][1]


def pending():
    me = out.name
    if args.get('wait_for', 'all') == 'all':
        names = [p.name for p in (queue / 'claims').iterdir() if p.name != me and not p.name.startswith(('eval', 'bench'))]
    else:
        names = args['wait_for'].split(',')
    return [n for n in names if not settled(n)]


while pending() and datetime.datetime.now() < until:
    print(f'{datetime.datetime.now():%H:%M} waiting for {pending()}', flush=True)
    time.sleep(60)
env = {**os.environ, 'PYTHONPATH': snapshot}
cache = queue / 'eval_cache.per_slice.npz'
for control, name in (('control', 'eval.json'), ('control_short', 'eval_vs_short.json'), ('control_10ep', 'eval_vs_10ep.json'),
                      ('full_control', 'eval_vs_full.json')):
    rc = subprocess.call([sys.executable, f'{snapshot}/scripts/eval/varnet_train_speed_eval.py', '--out', str(out / name),
                          '--control', control, '--cache', str(cache)], cwd=snapshot, env=env)
    print(f'eval --control {control} exit {rc}', flush=True)
print('Epoch = [   1/   1] TrainLoss = 0 ValLoss = 0 MeanSparsity = 0 TrainTime = 0s ValTime = 0s', flush=True)
