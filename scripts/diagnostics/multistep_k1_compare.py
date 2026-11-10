"""Bitwise comparison of two multistep_k1_reference.py outputs: loss, output, mask and every gradient.

Usage: python scripts/diagnostics/multistep_k1_compare.py <main.pt> <branch.pt>
"""
import sys
import torch

a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
ok = True
for name in a:
    for key in ('loss', 'output', 'mask'):
        same = torch.equal(a[name][key], b[name][key])
        ok &= same
        if not same:
            print(name, key, 'DIFFERS', (a[name][key] - b[name][key]).abs().max().item())
    if set(a[name]['grads']) != set(b[name]['grads']):
        ok = False
        print(name, 'grad keys differ', set(a[name]['grads']) ^ set(b[name]['grads']))
    for key in a[name]['grads']:
        if key in b[name]['grads'] and not torch.equal(a[name]['grads'][key], b[name]['grads'][key]):
            ok = False
            print(name, 'grad', key, 'DIFFERS', (a[name]['grads'][key] - b[name]['grads'][key]).abs().max().item())
    print(name, 'loss', a[name]['loss'].item(), 'grads', len(a[name]['grads']))
print('BITWISE EQUAL' if ok else 'MISMATCH')
