"""Final-epoch validation metrics of offline wandb runs, with each arm's difference from a control.

Reads `<run_dir>/wandb/latest-run/run-*.wandb` (no network, no GPU), takes the last epoch that logged `val/PSNR_Masked`
and writes `metrics.json` and `table.md` to --out. Per-anatomy values are the trainer's plain (unmasked) brain and knee
PSNR/NMSE/SSIM; PSNR_Masked/SSIM_Masked are pooled over the validation set under the apodized per-anatomy metric mask.
The trainer does not log knee-PD and knee-PDFS separately.

    python scripts/eval/wandb_final_val_metrics.py --control pre_network --out reports/<topic> \
        --run pre_network=<dir> --run null=<dir> --run post_network=<dir>
"""
import argparse
import glob
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

KEYS = ['val/PSNR_Masked', 'val/SSIM_Masked', 'val/NMSE', 'val/PSNR', 'val/SSIM',
        'val/brain/PSNR', 'val/brain/NMSE', 'val/brain/SSIM', 'val/knee/PSNR', 'val/knee/NMSE', 'val/knee/SSIM']


def last_validation(run_dir: str) -> tuple[int, dict]:
    """(number of validated epochs, the last validated epoch's KEYS)."""
    store = datastore.DataStore()
    store.open_for_scan(glob.glob(f'{run_dir}/wandb/latest-run/run-*.wandb')[0])
    rows = []
    while (data := store.scan_data()) is not None:
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof('record_type') == 'history':
            rows.append({item.key or '/'.join(item.nested_key): item.value_json for item in record.history.item})
    validated = [row for row in rows if 'val/PSNR_Masked' in row]
    return len(validated), {key: float(validated[-1][key]) for key in KEYS if key in validated[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, metavar='ARM=DIR')
    parser.add_argument('--control', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    runs = dict(item.split('=', 1) for item in args.run)
    results = {}
    for arm, run_dir in runs.items():
        epochs, metrics = last_validation(run_dir)
        results[arm] = {'validated_epochs': epochs, **metrics}
    control = results[args.control]
    for arm, metrics in results.items():
        metrics['delta_vs_control'] = {key: metrics[key] - control[key] for key in KEYS if key in metrics}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(results, indent=1))

    columns = [('PSNR_Masked', 'val/PSNR_Masked', 2), ('SSIM_Masked', 'val/SSIM_Masked', 4), ('NMSE', 'val/NMSE', 4),
               ('brain PSNR', 'val/brain/PSNR', 2), ('knee PSNR', 'val/knee/PSNR', 2),
               ('brain NMSE', 'val/brain/NMSE', 4), ('knee NMSE', 'val/knee/NMSE', 4)]
    lines = ['| arm | epochs | ' + ' | '.join(name for name, _, _ in columns) + ' |', '|' + '---|' * (len(columns) + 2)]
    for arm, metrics in results.items():
        lines.append(f"| {arm} | {metrics['validated_epochs']} | "
                     + ' | '.join(f'{metrics[key]:.{digits}f}' for _, key, digits in columns) + ' |')
    lines += ['', f'Difference from `{args.control}` (arm minus control; negative NMSE is better):', '',
              '| arm | ' + ' | '.join(name for name, _, _ in columns) + ' |', '|' + '---|' * (len(columns) + 1)]
    for arm, metrics in results.items():
        if arm != args.control:
            lines.append(f'| {arm} | ' + ' | '.join(f"{metrics['delta_vs_control'][key]:+.{digits}f}"
                                                   for _, key, digits in columns) + ' |')
    (out / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
