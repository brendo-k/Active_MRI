"""Markdown tables for the adaptive-sampler next steps from eval_adaptive_samplers_r8.py's output.

    python scripts/eval/summarize_adaptive_samplers_r8.py --tag plain
"""
import argparse
import json
from pathlib import Path

OUT = Path('reports/adaptivity_next_steps_r8')
GENERALIST_HISTORY = Path('/tmp/bkadota_runs/generalist_vd_varnet_r8/history.json')
DESCRIPTIONS = {  # run: (sampler, reconstructor)
    'loupe_original': ('LOUPE (start point)', 'LOUPE run, co-adapted'),
    'static_joint_R8': ('static (offset off)', 'joint fine-tune'),
    'adapt16_joint_R8': ('adaptive head, 16 x 16 grid', 'joint fine-tune'),
    'anattable_joint_R8': ('one map per anatomy', 'joint fine-tune'),
    'adapt16_frozen-loupe-recon_R8': ('adaptive head, 16 x 16 grid', 'LOUPE run, frozen'),
    'adapt16_frozen-generalist_R8': ('adaptive head, 16 x 16 grid', 'generalist, frozen'),
    'static_frozen-generalist_R8': ('static (offset off)', 'generalist, frozen'),
    'static_joint_flags_R8': ('static (offset off)', 'flagged VarNet, joint fine-tune'),
    'adapt16_joint_flags_R8': ('adaptive head, 16 x 16 grid', 'flagged VarNet, joint fine-tune'),
    'anattable_joint_flags_R8': ('one map per anatomy', 'flagged VarNet, joint fine-tune'),
}


def ci(entry, digits=3):
    return f"{entry['mean']:+.{digits}f} [{entry['ci95'][0]:+.{digits}f}, {entry['ci95'][1]:+.{digits}f}]"


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
                     + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', default='plain')
    args = parser.parse_args()
    results = json.loads((OUT / f'eval_{args.tag}.json').read_text())
    paired = results.pop('paired', {})
    lines = []
    if GENERALIST_HISTORY.exists():
        history = json.loads(GENERALIST_HISTORY.read_text())
        final = history['history'][-1]
        reference = history['reference']
        lines.append('Generalist reconstructor (step %d) vs the LOUPE run\'s own, 86 val slices, PSNR dB:\n' % final['step'])
        lines.append(table(['masks', 'LOUPE run recon', 'generalist', 'difference'],
                           [[k, f"{reference[k]:.2f}", f"{final[k]:.2f}", f"{final[k] - reference[k]:+.2f}"]
                            for k in ('loupe', 'conv', 'loupe_reshaped')]))
        lines.append('')
    rows = []
    for name, r in results.items():
        sampler, recon = DESCRIPTIONS.get(name, (name, ''))
        offset = (f"{r['offset_mean_abs']:.3f} / {r['offset_across_slice_std']:.3f}" if 'offset_mean_abs' in r else '')
        rows.append([name, sampler, recon, f"{r['psnr']['brain']:.3f} / {r['psnr']['knee']:.3f}",
                     ci(r['anatomy_swap_cost_db']), ci(r['slice_swap_cost_db']),
                     f"{r['slice_vs_mean_corr']['brain']:.4f} / {r['slice_vs_mean_corr']['knee']:.4f}",
                     f"{r['brain_vs_knee_corr']:.4f}", offset])
    lines.append(table(['run', 'sampler', 'reconstructor', 'val PSNR brain / knee', 'anatomy swap cost (dB)',
                        'slice swap cost (dB)', 'slice vs mean corr brain / knee', 'brain vs knee corr',
                        'offset mean abs / across-slice std (logits)'], rows))
    lines.append('')
    if paired:
        lines.append(table(['comparison (per-slice PSNR, dB)', 'all', 'brain', 'knee'],
                           [[k, ci(v['all']), ci(v['brain']), ci(v['knee'])] for k, v in paired.items()]))
    text = '\n'.join(lines) + '\n'
    (OUT / f'summary_{args.tag}.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
