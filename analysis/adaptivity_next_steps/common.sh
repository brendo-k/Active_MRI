# Shared settings for the adaptivity next-step sampler runs (plain VarNet; see reports/ADAPTIVITY_MECHANISMS_R8.md).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
LOUPE=checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05/last_model.pt
GENERALIST=/tmp/bkadota_runs/generalist_vd_varnet_r8/last_model.pt
RUNS=/tmp/bkadota_runs
LOG=analysis/adaptivity_next_steps
# One run: name, reconstructor checkpoint, then extra overrides. Every arm starts from LOUPE's mask.
sampler_run() {
  name=$1; recon=$2; shift 2
  echo "start $name $(date)" >> $LOG/launcher.log
  python train.py --model sequential --recon varnet --dataset brain_knee_balanced --acceleration 8 --batch-size 16 \
    --epochs ${EPOCHS:-10} --lr 1e-4 --seed 42 --out-dir $RUNS/$name --pretrained-recon $recon \
    sampler.sampler_type=smooth_adapt_2d sampler.slope=5 sampler.mask_grad_floor=0.05 sampler.ste_grad_floor=0.05 \
    sampler.init_mask_checkpoint=${INIT_MASK:-$LOUPE} sampler.lr=1e-3 training.lr_warmup_epochs=0 training.lr_step_size=1000 \
    output.run_name=$name "$@" > $LOG/$name.log 2>&1
  echo "done $name $(date) rc=$?" >> $LOG/launcher.log
}
