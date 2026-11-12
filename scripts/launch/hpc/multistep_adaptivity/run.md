# multistep_adaptivity on CoreHPC

The frozen plan is `reports/multistep_adaptivity/proposal.md`; the phases are in its section 8 and the platform rules in
section 9. Everything below that computes runs as a SLURM job. The login node only receives files (rsync) and answers
`sinfo`, `squeue` and `ls`. `$EXP_DIR` is `~/scratch/multistep_adaptivity` on CoreHPC.

## 0. Pin the GPU type, once

```bash
sinfo -p gpu -o "%G %N %t" | sort -u        # login node, read-only
```

Pick H100 NVL or RTX Pro 6000 (at least 80 GB; L40S is excluded). Write its gres string and the name
`torch.cuda.get_device_name()` gives into `reports/multistep_adaptivity/p0/README.md`. Every job gets
`--gres=gpu:<type>:1` and `GPU_NAME=<that name or a unique part>`, and neither changes after P1 starts. A job on another
card exits before it starts.

## 1. Ship the code from the lab server

A shallow clone keeps `.git`, so each run's `run_info.json` records the real commit. Nothing runs git on the login node.

```bash
ship="$(mktemp -d)"
git clone -q --depth 1 --branch exp/multistep_adaptivity file:///home/bkadota/code/active_mri/SeqMRI "$ship/code"
git -C "$ship/code" rev-parse HEAD           # the commit you ship: EXPECTED_COMMIT below
rsync -a --delete "$ship/code/" corehpc-login:scratch/multistep_adaptivity/code/
ssh corehpc-login 'mkdir -p scratch/multistep_adaptivity/{slurm_log,runs,scores}'
```

`corehpc-login` stands for `chpc-ucsf-login-vm1.corehpc.ucsf.edu`, reached through `chpc-ucsf-bastion-vm1.corehpc.ucsf.edu`.
Ship again the same way after any new commit; runs already started keep the commit they recorded.

## 2. Validation job (P0 item 7, blocks P1)

```bash
cd ~/scratch/multistep_adaptivity
sbatch --gres=gpu:<type>:1 --export=ALL,GPU_NAME='<name>',EXPECTED_COMMIT=<hash> \
    code/scripts/launch/hpc/multistep_adaptivity/validate.sbatch
```

The log `slurm_log/multistep_validate-<job>.log` ends with a pass/FAIL table, and each check's own log is in
`validate/<job>/`. Every check must pass. Copy the logs back into `reports/multistep_adaptivity/p0/corehpc/`.

## 3. P1: bases, then the pilot

```bash
J='--gres=gpu:<type>:1'; C=code/scripts/launch/hpc/multistep_adaptivity
for s in 42 43; do sbatch $J --time=1-00:00:00 --export=ALL,ARM=B_$s,GPU_NAME='<name>' $C/train.sbatch; done
# after B_42 finishes (its scores are in scores/B_42.json): the pilot is the real seed-42 K=16 pair
for arm in A_16_s42 T_16_s42; do sbatch $J --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME='<name>' $C/train.sbatch; done
# once both have runs/<arm>/epoch_010.pt: the gate, scored by the same eval as the final results
for arm in A_16_s42 T_16_s42; do sbatch $J --export=ALL,ARM=$arm,CHECKPOINT=epoch_010.pt,GPU_NAME='<name>' $C/eval.sbatch; done
```

The gates (stability, plumbing, head used) are read from `scores/{A,T}_16_s42_epoch_010.json` and `scores/B_42.json`. The
AGENTS.md roadmap line changes on the branch when P1 starts.

## 4. P2: the rest of the sweep, the floor and the ablations

```bash
for s in 42 43; do for k in 1 4 16; do for a in A T; do
    [[ $s == 42 && $k == 16 ]] && continue                     # the pilot pair is already running
    sbatch $J --time=1-00:00:00 --export=ALL,ARM=${a}_${k}_s$s,GPU_NAME='<name>' $C/train.sbatch
done; done; done
for arm in T_1_s42_f142 T_16_s42_f142 A_4_s42_rec A_16_s42_img; do
    sbatch $J --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME='<name>' $C/train.sbatch
done
```

## 5b. Knee-only from-scratch subproblem (replaces P2b, `reports/multistep_adaptivity/addendum_2026-11-11.md`)

Iteration n is defined in `common.sh` as `_kscr<n>`; an undefined n refuses to start. Each iteration is an A/T pair and a
floor twin, validated and scored on the dev half of the knee val volumes:

```bash
for arm in A_16_s42_kscr1 T_16_s42_kscr1 T_16_s142_kscr1; do
    sbatch $J --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME='<name>' $C/train.sbatch
done
```

The holdout half is scored only for the confirmation, with `eval_multistep_adaptivity.py score ... --val-half holdout`.

## 5. P2b as frozen (VD p=4 start): replaced by 5b on 11-11, not run

```bash
for arm in A_16_s42_scr T_16_s42_scr; do sbatch $J --time=2-00:00:00 --export=ALL,ARM=$arm,GPU_NAME='<name>' $C/train.sbatch; done
```

## 6. Bring the results back

```bash
rsync -a corehpc-login:scratch/multistep_adaptivity/{scores,wandb,slurm_log} <lab-server folder>/
wandb sync <lab-server folder>/wandb/wandb/offline-run-*     # on the lab server, which has internet
python scripts/eval/eval_multistep_adaptivity.py contrast --scores <lab-server folder>/scores \
    --pair G_16_s42=A_16_s42-T_16_s42 ... --out reports/multistep_adaptivity/contrasts.json
```

A crashed run is rerun in full: move `runs/<arm>` aside and submit it again. `train.sbatch` will not start in a run
directory that already holds a checkpoint. Checkpoints stay on CoreHPC scratch.
