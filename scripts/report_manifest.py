"""Provenance manifest for a self-contained report folder, and the conformance check the experiment-report skill runs.

A report lives in `reports/<name>/`:

    report.md       the write-up; starts with the provenance block that `write` prints, then the skill's headings
    manifest.json   code commit, metric mask, data, runs (overrides, config hash, wandb id, checkpoint), commands, results
    run.sh          the commands that regenerate every result file
    <results>       the json/npz/png files the report cites, next to it (flat, or in results/ and figures/)

    python scripts/report_manifest.py write reports/metric_mask_reeval_r8 --metric-mask apodized \\
        --runs 'loupe2d_cgsense_r8_*' --script scripts/eval/reeval_metric_mask_r8.py --produced-at 4f70ea6
    python scripts/report_manifest.py check reports/metric_mask_reeval_r8
"""
import argparse
import datetime
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / 'checkpoints'
NOT_RESULTS = {'report.md', 'manifest.json', 'run.sh'}
REQUIRED_HEADINGS = ('Question', 'Runs compared', 'Methods', 'Results', 'Conclusions', 'Caveats', 'Figures',
                     'Next steps', 'Reproduce')
METRIC_MASKS = {
    'apodized': 'k-space-apodized per-anatomy mask (activemri/utils/metric_mask.py), the trainer metric since 2026-10-20',
    'legacy': 'legacy mask, target > 0.1 * max of the RSS target; not comparable to apodized-mask numbers',
}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.rstrip('\n')


def sha256(path, limit=None):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        remaining = limit
        while chunk := handle.read(1 << 20 if remaining is None else min(1 << 20, remaining)):
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return digest.hexdigest()


def metric_mask_description(version):
    description = {'version': version, 'definition': METRIC_MASKS[version]}
    if version == 'apodized':
        sys.path.insert(0, str(ROOT))
        from activemri.utils.metric_mask import APODIZATION_SIGMA, MASK_PARAMS
        description['apodization_sigma'] = APODIZATION_SIGMA
        description['params'] = MASK_PARAMS
    return description


def describe_run(run_dir, hash_checkpoints):
    hydra_dir = run_dir / '.hydra'
    config_path = hydra_dir / 'config.yaml'
    entry = {'name': run_dir.name, 'path': str(run_dir.relative_to(ROOT))}
    if config_path.exists():
        entry['config_sha256'] = sha256(config_path)[:16]
    overrides = hydra_dir / 'overrides.yaml'
    if overrides.exists():
        entry['overrides'] = yaml.safe_load(overrides.read_text())
    wandb_runs = sorted((run_dir / 'wandb').glob('run-*')) if (run_dir / 'wandb').exists() else []
    if wandb_runs:
        entry['wandb_run_id'] = wandb_runs[-1].name.rsplit('-', 1)[-1]
    run_info = run_dir / 'run_info.json'
    entry['trained_at_commit'] = json.loads(run_info.read_text()).get('commit') if run_info.exists() else None
    checkpoints = {}
    for name in ('best_model.pt', 'last_model.pt'):
        path = run_dir / name
        if path.exists():
            info = {'bytes': path.stat().st_size,
                    'modified': datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat(timespec='seconds')}
            if hash_checkpoints:
                info['sha256'] = sha256(path)[:16]
            checkpoints[name] = info
    entry['checkpoints'] = checkpoints
    return entry


def dataset_description(run_dirs):
    for run_dir in run_dirs:
        config_path = run_dir / '.hydra' / 'config.yaml'
        if config_path.exists():
            return yaml.safe_load(config_path.read_text()).get('datasets')
    return None


def result_files(report_dir):
    return sorted(path for path in report_dir.rglob('*') if path.is_file() and path.name not in NOT_RESULTS)


def write(args):
    report_dir = (ROOT / args.report_dir).resolve() if not Path(args.report_dir).is_absolute() else Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.no_runs and args.runs:
        raise SystemExit('--no-runs and --runs are mutually exclusive')
    produced_at = git('rev-parse', args.produced_at)
    run_dirs = [] if args.no_runs else sorted(
        path for path in CHECKPOINT_ROOT.iterdir()
        if path.is_dir() and any(fnmatch.fnmatch(path.name, pattern) for pattern in args.runs))
    changed_since = set(git('diff', '--name-only', produced_at, 'HEAD').split())
    manifest = {
        'schema': 1,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'code': {
            'produced_at': produced_at,
            'produced_at_subject': git('log', '-1', '--format=%s', produced_at),
            'head': git('rev-parse', 'HEAD'),
            'branch': git('branch', '--show-current'),
            'dirty_tracked_files': [line[3:] for line in git('status', '--short', '--untracked-files=no').splitlines()
                                if not (ROOT / line[3:].split(' -> ')[-1]).resolve().is_relative_to(report_dir)],
            'scripts': [{'path': script, 'changed_since_produced_at': script in changed_since} for script in args.script],
        },
        'metric_mask': metric_mask_description(args.metric_mask),
        'data': {'config': dataset_description(run_dirs), 'note': args.data_note},
        'runs': [describe_run(run_dir, args.hash_checkpoints) for run_dir in run_dirs],
        'no_runs': args.no_runs,
        'commands': args.command,
        'results': [{'path': str(path.relative_to(report_dir)), 'bytes': path.stat().st_size, 'sha256': sha256(path)[:16]}
                    for path in result_files(report_dir)],
    }
    (report_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(provenance_block(manifest))


def provenance_block(manifest):
    code = manifest['code']
    dirty = code['dirty_tracked_files']
    stale = [script['path'] for script in code['scripts'] if script['changed_since_produced_at']]
    untrained = [run['name'] for run in manifest['runs'] if not run.get('trained_at_commit')]
    lines = [
        '| | |', '|---|---|',
        f"| Code | results produced at `{code['produced_at'][:7]}` ({code['produced_at_subject']}); branch `{code['branch']}` |",
        f"| Metric mask | **{manifest['metric_mask']['version']}**: {manifest['metric_mask']['definition']} |",
        ('| Runs | none: this report loads no checkpoint |' if manifest.get('no_runs') else
         f"| Runs | {len(manifest['runs'])} checkpoint directories under `checkpoints/` (untracked; names, overrides, wandb ids in `manifest.json`) |"),
        f"| Results | {len(manifest['results'])} files next to this report, hashed in `manifest.json` |",
        '| Reproduce | `bash run.sh` from the repo root |',
    ]
    if stale:
        lines.append(f"| Code drift | changed after the results were produced: {', '.join(f'`{path}`' for path in stale)} |")
    if dirty:
        lines.append(f"| Working tree | {len(dirty)} tracked files were modified when the manifest was written |")
    if untrained:
        lines.append('| Training commit | not recorded for these runs (they predate `run_info.json`) |')
    return '\n'.join(lines)


def check(args):
    report_dir = Path(args.report_dir)
    report_dir = report_dir if report_dir.is_absolute() else ROOT / report_dir
    problems = []

    def need(condition, message):
        if not condition:
            problems.append(message)

    manifest_path, report_path, run_path = report_dir / 'manifest.json', report_dir / 'report.md', report_dir / 'run.sh'
    need(report_path.exists(), 'report.md missing')
    need(run_path.exists(), 'run.sh missing')
    need(manifest_path.exists(), 'manifest.json missing (run `write`)')
    if problems:
        return finish(report_dir, problems)
    manifest = json.loads(manifest_path.read_text())
    text = report_path.read_text()
    headings = {line.lstrip('#').strip() for line in text.splitlines() if line.startswith('#')}
    for heading in REQUIRED_HEADINGS:
        need(any(h.lower().startswith(heading.lower()) for h in headings), f'report.md has no "{heading}" heading')
    need(manifest['code']['produced_at'][:7] in text, 'report.md does not cite the commit the results were produced at')
    need(f"**{manifest['metric_mask']['version']}**" in text, 'report.md provenance block does not state the metric mask version')
    try:
        git('cat-file', '-e', manifest['code']['produced_at'])
    except subprocess.CalledProcessError:
        problems.append('produced_at commit is not in this repository')
    for script in manifest['code']['scripts']:
        need((ROOT / script['path']).exists(), f"script {script['path']} does not exist")
        need(script['path'] in run_path.read_text(), f"run.sh does not call {script['path']}")
        need(script['path'] in text, f"report.md does not cite {script['path']}")
    for command in manifest['commands']:
        need(command in run_path.read_text(), f'run.sh is missing the manifest command: {command[:60]}')
    need(manifest['runs'] or manifest.get('no_runs'), 'manifest lists no runs (pass --no-runs to `write` if the report loads no checkpoint)')
    for result in manifest['results']:
        path = report_dir / result['path']
        need(path.exists(), f"result {result['path']} is missing")
        if path.exists():
            need(sha256(path)[:16] == result['sha256'], f"result {result['path']} changed since the manifest was written")
    listed = {result['path'] for result in manifest['results']}
    for path in result_files(report_dir):
        need(str(path.relative_to(report_dir)) in listed, f"{path.relative_to(report_dir)} is not in the manifest (rerun `write`)")
    return finish(report_dir, problems)


def finish(report_dir, problems):
    name = report_dir.relative_to(ROOT) if report_dir.is_relative_to(ROOT) else report_dir
    if problems:
        print(f'{name}: {len(problems)} problem(s)')
        for problem in problems:
            print(f'  - {problem}')
        return 1
    print(f'{name}: conforms')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    writer = sub.add_parser('write', help='write manifest.json and print the provenance block for report.md')
    writer.add_argument('report_dir')
    writer.add_argument('--metric-mask', choices=sorted(METRIC_MASKS), required=True)
    writer.add_argument('--runs', nargs='+', default=[], help='checkpoint directory names or globs the report uses')
    writer.add_argument('--no-runs', action='store_true', help='the report loads no checkpoint (training-free analysis or synthesis)')
    writer.add_argument('--script', nargs='+', default=[], help='scripts that produced the results')
    writer.add_argument('--command', action='append', default=[], help='a command run.sh executes (repeatable)')
    writer.add_argument('--produced-at', default='HEAD', help='commit whose code produced the results')
    writer.add_argument('--data-note', default='', help='split, volumes and slices when not obvious from the run config')
    writer.add_argument('--hash-checkpoints', action='store_true', help='sha256 every checkpoint (slow for large ones)')
    checker = sub.add_parser('check', help='verify a report folder follows the convention; exit 1 if not')
    checker.add_argument('report_dir')
    args = parser.parse_args()
    if args.action == 'write':
        write(args)
        return 0
    return check(args)


if __name__ == '__main__':
    sys.exit(main())
