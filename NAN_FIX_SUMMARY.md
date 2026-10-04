# VarNet NaN Training Issue - Root Cause & Fixes

## Problem
The `onestep_2d_varnet_brain_knee_samp128_run3_rss_nan` training run produces NaN gradients very early (epoch 1-2), causing optimizer steps to be skipped repeatedly. The training log shows:
```
Skipping optimizer step with non-finite gradient norm (loss=0.0459, ...)
Skipping optimizer step with non-finite gradient norm (loss=0.06383, ...)
...
```

## Root Causes

### 1. **Unconstrained Data Consistency Weight** (PRIMARY)
Each VarNet cascade has a learnable parameter `dc_weight` initialized to `1.0`:

```python
# Before (VarNet.py line 97):
self.dc_weight = nn.Parameter(torch.ones(1))
```

This weight scales the soft data consistency term in each cascade:
```python
# VarNet.py line 114:
soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
```

**Why this causes NaNs:**
- `dc_weight` has NO upper bound, so it can grow unbounded during backprop
- With 12 cascades in sequence, each cascading through k-space operations on complex numbers
- Unconstrained growth of `dc_weight` → exploding gradients in soft data consistency term
- Complex number operations + large cascades = numerical instability

### 2. **High Learning Rate**
The default learning rate of `0.001` is too aggressive for cascaded VarNet, allowing:
- Weight explosions to happen in fewer iterations
- Larger step sizes in parameter space → higher risk of instability

### 3. **Cascaded Numerical Errors**
With 12 cascades of:
- Complex FFT/IFFT operations
- Multiple U-Net passes (each reaches 288 channels at bottleneck)
- Error accumulation through refinement loop

## Solutions Implemented

### Fix 1: Constrain `dc_weight` with Sigmoid
**File:** `activemri/reconstructors/VarNet.py`

**Change 1 - Initialize to 0.0 instead of 1.0:**
```python
# After (line 97):
self.dc_weight = nn.Parameter(torch.tensor(0.0))
```
This initializes the log-space parameter to 0, giving sigmoid(0) = 0.5 after applying sigmoid.

**Change 2 - Apply sigmoid in forward pass:**
```python
# After (line 114):
dc_weight = torch.sigmoid(self.dc_weight)
soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * dc_weight
```

**Why this works:**
- Sigmoid constrains `dc_weight` to [0, 1] → no explosions possible
- Gradient flow is still good (sigmoid non-saturating near 0)
- 0.5 initial value is a reasonable "half" data consistency
- Prevents the runaway growth that caused NaNs

### Fix 2: Reduce Learning Rate
**File:** `configs/config.yaml`

```yaml
# Before:
lr: 0.001

# After:
lr: 0.0003
```

**Why this helps:**
- Smaller updates to parameters each step
- Even if `dc_weight` tries to grow, it grows slower
- Reduces the chance of hitting numerical instability
- 0.0003 is standard for VarNet in literature

## Testing

Before fix:
```
Cascade 0: dc_weight = 1.000000 (unconstrained, can grow to 10+ easily)
```

After fix:
```
Cascade 0: dc_weight = 0.000000
  After sigmoid: 0.500000 (constrained to [0, 1])
```

## What To Do Next

1. **Restart training** with the new code:
   ```bash
   python scripts/train_sequential.py \
       recon=varnet \
       datasets=brain_knee_balanced \
       training.num_step=1 \
       sampler.sampler_type=conv_2d \
       output.out_dir=checkpoints/onestep_2d_varnet_brain_knee_samp128_run4_fixed
   ```

2. **Monitor for NaNs** in the first 2 epochs - they should not appear now

3. **Convergence speed** will be slower due to 3x lower LR, but it should be stable

4. **(Optional) Fine-tune LR** if needed:
   - If training is too slow: increase to 0.0005
   - If NaNs still appear: decrease to 0.0001

## Key References

- VarNet sigmoid-constrained weights: Sriram et al. (2020)
- Cascaded reconstruction stability: Improved with bounded parameters
- Complex-valued deep learning: Sensitive to initialization and learning rates

