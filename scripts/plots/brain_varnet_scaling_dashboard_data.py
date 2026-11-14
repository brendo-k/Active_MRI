"""Data files for the brain_varnet_scaling dashboard, from reports/brain_varnet_scaling/eval/summary.json and the
runs' train_sequential.log. Each file is JSON {rows, meta}; meta carries this script's text.

  runs.json         one row per run: full-val PSNR / SSIM / NMSE at best_model.pt and last_model.pt, train - val gaps,
                    trainer-val peak, where training stopped
  curves.json       one row per epoch: train loss, and trainer brain val loss / PSNR / SSIM / NMSE where validated
                    (150 slices)
  six_vs_twelve.json  6 minus 12 cascades per data size, seed and checkpoint, paired per slice, 95% CI over volumes
  cost.json         parameters and seconds per optimiser step per cascade count

    python scripts/plots/brain_varnet_scaling_dashboard_data.py --out <folder>
"""
import argparse
import json
import re
from pathlib import Path

QUEUE = Path('/home/bkadota/data/seqmri_checkpoints/brain_varnet_scaling')
SUMMARY = Path('reports/brain_varnet_scaling/eval/summary.json')
STEPS_PER_EPOCH = {20: 19, 40: 38, 80: 76}
SLICES = {20: 301, 40: 597, 80: 1211}
EPOCH = re.compile(r'Epoch = \[\s*(\d+)/\s*(\d+)\] TrainLoss = ([\d.e+-]+) ValLoss = ([\w.e+-]+)')
BRAIN = re.compile(r'\[brain\][^\n]*?NMSE=([\d.e+-]+) PSNR=([\d.]+) SSIM=([\d.]+)')
EARLY = re.compile(r'Early stop after epoch (\d+)')


def parse_log(log):
    rows, pending, early = [], None, None
    for line in log.read_text().splitlines():
        if (m := BRAIN.search(line)):
            pending = {'val_nmse': float(m[1]), 'val_psnr': float(m[2]), 'val_ssim': float(m[3])}
        elif (m := EPOCH.search(line)):
            val_loss = None if m[4] == 'nan' else float(m[4])
            rows.append({'epoch': int(m[1]), 'train_loss': float(m[3]), 'val_loss': val_loss,
                         **(pending or {'val_nmse': None, 'val_psnr': None, 'val_ssim': None})})
            pending = None
        elif (m := EARLY.search(line)):
            early = int(m[1])
    return rows, early


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--summary', type=Path, default=SUMMARY)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    meta = [{'label': 'scripts/plots/brain_varnet_scaling_dashboard_data.py', 'value': Path(__file__).read_text(),
             'type': 'code', 'lang': 'python'},
            {'label': 'Input', 'value': f'{args.summary} and {args.queue}/c*_v*_s*/train_sequential.log'}]

    runs, curves = [], []
    for r in summary['runs']:
        c, v, s = r['cascades'], r['volumes'], r['seed']
        name = f'c{c}_v{v}_s{s}'
        log, early = parse_log(args.queue / name / 'train_sequential.log')
        spe = STEPS_PER_EPOCH[v]
        validated = [e for e in log if e['val_psnr'] is not None]
        peak = max(validated, key=lambda e: e['val_psnr'])
        runs.append({'run': name, 'cascades': c, 'volumes': v, 'train_slices': SLICES[v], 'seed': s,
                     'best_psnr': round(r['best_psnr'], 3), 'best_ssim': round(r['best_ssim'], 4),
                     'best_nmse': round(r['best_nmse'], 5), 'last_psnr': round(r['last_psnr'], 3),
                     'last_ssim': round(r['last_ssim'], 4), 'last_nmse': round(r['last_nmse'], 5),
                     'gap_best': round(r['gap_best'], 3), 'gap_last': round(r['gap_last'], 3),
                     'trainer_val_peak_psnr': peak['val_psnr'], 'trainer_val_peak_step': peak['epoch'] * spe,
                     'stopped_at_step': log[-1]['epoch'] * spe, 'early_stopped': early is not None,
                     'early_stopping_on': name not in ('c12_v40_s42', 'c12_v80_s42')})
        for e in log:
            curves.append({'run': name, 'cascades': c, 'volumes': v, 'seed': s, 'step': e['epoch'] * spe,
                           'train_loss': e['train_loss'], 'val_loss': e['val_loss'], 'val_psnr': e['val_psnr'],
                           'val_ssim': e['val_ssim'], 'val_nmse': e['val_nmse']})

    pairs = []
    for key, d in summary['c6_minus_c12'].items():
        m = re.match(r'v(\d+)_(s\d+|two_seed)_(best_model|last_model)', key)
        pairs.append({'comparison': key, 'volumes': int(m[1]),
                      'seed': 'mean of 42 and 43' if m[2] == 'two_seed' else m[2][1:],
                      'checkpoint': 'peak' if m[3] == 'best_model' else 'last',
                      'diff_db': round(d['diff'], 3), 'ci_low': round(d['ci'][0], 3), 'ci_high': round(d['ci'][1], 3)})
    cost = [{'cascades': int(k[1:]), 'parameters_m': round(v['parameters'] / 1e6, 2),
             'seconds_per_step': round(v['seconds_per_step'], 3)} for k, v in summary['cost'].items()]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('runs', runs), ('curves', curves), ('six_vs_twelve', pairs), ('cost', cost)):
        (args.out / f'{name}.json').write_text(json.dumps({'rows': rows, 'meta': meta}, separators=(',', ':')))
        print(name, len(rows), (args.out / f'{name}.json').stat().st_size)


if __name__ == '__main__':
    main()
