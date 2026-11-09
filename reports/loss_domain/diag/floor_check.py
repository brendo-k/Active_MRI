import sys
from pathlib import Path
import torch
d = torch.load(Path(sys.argv[1]) / 'cross_loss_examples.pt', weights_only=False)
print(f"{'cond':10s} {'target floor':>13s} {'t-ksp inside':>13s} {'t-rss inside':>13s} {'t-|xE| inside':>14s} {'bg share of grid':>17s}")
for i, c in enumerate(d['rss']['cond']):
    t, o_rss, o_k, xe = d['rss']['t'][i], d['rss']['out'][i], d['kspace_rss']['out'][i], d['rss']['xe'][i].abs()
    background = o_rss < 1e-3
    inside = (o_rss > 0.3 * o_rss.max())
    print(f"{c:10s} {t[background].median():13.3f} {(t - o_k)[inside].median():13.3f} {(t - o_rss)[inside].median():13.3f} "
          f"{(t - xe)[inside].median():14.3f} {background.float().mean():17.2f}")
