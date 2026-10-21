"""Relate the per-slice PSNR difference (bilinear - conv) to sense-map diagnostics and list the extreme slices.

Reads reports/sens_upsample_r8/sense_maps_per_slice.npz (from sens_upsample_sense_maps_r8.py).

    python scripts/diagnostics/sens_upsample_sense_maps_summary_r8.py
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path('reports/sens_upsample_r8')
EXTREME = 10


def main():
    d = np.load(OUT_DIR / 'sense_maps_per_slice.npz')
    delta = d['bilinear_psnr'] - d['conv_psnr']
    drivers = {'conv_psnr': d['conv_psnr'], 'map_rel_diff': d['map_rel_diff'],
               'fit_delta': d['bilinear_fit'] - d['conv_fit'], 'oracle_nmse_delta': d['bilinear_oracle_nmse'] - d['conv_oracle_nmse'],
               'high_frequency_delta': d['bilinear_high_frequency'] - d['conv_high_frequency'],
               'oracle_nmse_conv': d['conv_oracle_nmse'], 'fit_conv': d['conv_fit']}
    result = {'corr_with_psnr_delta': {}, 'extreme': {}}
    for anatomy in ('brain', 'knee'):
        rows = d['anatomy'] == anatomy
        result['corr_with_psnr_delta'][anatomy] = {k: float(np.corrcoef(v[rows], delta[rows])[0, 1]) for k, v in drivers.items()}
    # high-PSNR slices: top quintile of conv PSNR within knee
    knee = d['anatomy'] == 'knee'
    high = knee & (d['conv_psnr'] >= np.percentile(d['conv_psnr'][knee], 80))
    low = knee & (d['conv_psnr'] <= np.percentile(d['conv_psnr'][knee], 20))
    result['knee_psnr_quintiles'] = {name: {'n': int(r.sum()), 'psnr_delta_mean': float(delta[r].mean()), 'psnr_delta_sd': float(delta[r].std()),
                                            'conv_psnr_mean': float(d['conv_psnr'][r].mean()), 'oracle_nmse_delta_mean': float(drivers['oracle_nmse_delta'][r].mean()),
                                            'oracle_nmse_conv_mean': float(d['conv_oracle_nmse'][r].mean()), 'fit_conv_mean': float(d['conv_fit'][r].mean()),
                                            'map_rel_diff_mean': float(d['map_rel_diff'][r].mean())}
                                    for name, r in (('top20pct_conv_psnr', high), ('bottom20pct_conv_psnr', low))}
    order = np.argsort(delta)
    show = lambda i: {'volume': str(d['volume'][i]), 'slice': int(d['slice_id'][i]), 'delta_psnr': float(delta[i]), 'conv_psnr': float(d['conv_psnr'][i]),
                      'map_rel_diff': float(d['map_rel_diff'][i]), 'fit_delta': float(drivers['fit_delta'][i]),
                      'oracle_nmse_delta': float(drivers['oracle_nmse_delta'][i]), 'ssim_delta': float(d['bilinear_ssim'][i] - d['conv_ssim'][i])}
    result['extreme'] = {'bilinear_worse': [show(i) for i in order[:EXTREME]], 'bilinear_better': [show(i) for i in order[::-1][:EXTREME]]}
    (OUT_DIR / 'sense_maps_drivers.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('corr_with_psnr_delta', 'knee_psnr_quintiles')}, indent=1))
    for key, rows in result['extreme'].items():
        print(key)
        for r in rows:
            print('  ', r)


if __name__ == '__main__':
    main()
