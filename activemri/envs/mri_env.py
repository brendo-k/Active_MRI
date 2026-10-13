"""Step-wise (reset/step) RL environment for the DQN sampler.

`Sequential_Trainer`/`SequentialSampling` only support one differentiable
forward+backward per batch (straight-through or single-shot REINFORCE) --
there's no notion of a per-step reward or an off-policy replay buffer
anywhere in that path. Q-learning needs exactly that, so this wraps the same
dataset + `VarNet` reconstructor + `SensetivityModel_mc` used elsewhere in
this repo with `reset()`/`step(action)` semantics modeled on
`active-mri-acquisition`'s `activemri.envs.envs.ActiveMRIEnv`.

One discrete action = one k-space column (matching the DQN sampler's action
space); reward is the improvement in NMSE from adding that column.

`reward_mode="sparse"` follows L2SR (Learning to Sample and Reconstruct,
https://arxiv.org/abs/2212.02190): the reconstructor is only run at the start
and the end of an episode, every intermediate reward is 0, and the terminal
reward is the whole episode's NMSE improvement (the sum of what `"dense"`
would have paid out step by step). Since the observation then can't contain the
current reconstruction, it carries the zero-filled image instead.

`action_space="point"` switches from one k-space column per action to
individual 2D k-space points (all of H*W), with `points_per_step` points
acquired per `step()` so the horizon stays manageable, and a
`center_block` x `center_block` fully-sampled square preselected instead of
center columns.

`observation` picks what the policy sees besides the mask: `"kspace"` (the
default) is the coil-combined k-space, `"image"` is the SENSE-combined
zero-filled image of the multicoil data actually acquired so far.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch

from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.reconstructors.VarNet import VarNet
from activemri.utils.image_processing import fft_2d_img, ifft_2d_img


OBSERVATIONS = ("kspace", "kspace_aligned", "image", "image_kspace_aligned")
ROW_PHASE_MIN_GAIN = 5.0  # the sense maps' row phase must concentrate the DC energy this much more than not


def _nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-example NMSE, `[B]`."""
    return (
        (pred - target).abs().pow(2).flatten(1).sum(dim=1)
        / target.abs().pow(2).flatten(1).sum(dim=1).clamp(min=1e-12)
    )


def center_block_mask(batch_size: int, resolution: List[int], block: int, device: torch.device) -> torch.Tensor:
    """`[B, 1, H, W]` mask with the centered `block` x `block` square set."""
    h, w = resolution
    mask = torch.zeros(batch_size, 1, h, w, device=device)
    half = block // 2
    mask[:, :, h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half] = 1
    return mask


def encode_observation(mask: torch.Tensor, single_coil_k: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Packs (acquired k-space, `second`, mask) into `[B, 5, H, W]`.

    `single_coil_k` and `second` are complex `[B, H, W]`: the coil-combined full
    k-space and the k-space-or-image the policy sees beside the acquired samples.
    """
    acquired = mask.squeeze(1) * single_coil_k
    obs = torch.cat(
        [torch.view_as_real(acquired), torch.view_as_real(second)], dim=-1
    ).permute(0, 3, 1, 2)
    return torch.cat([obs, mask], dim=1)


def coil_combined_kspace(full_kspace_complex: torch.Tensor, sense_maps: torch.Tensor) -> torch.Tensor:
    """Complex `[B, H, W]` single-coil k-space of the sense-weighted coil combination."""
    return fft_2d_img((ifft_2d_img(full_kspace_complex) * sense_maps.conj()).sum(dim=1))


class MRIEnv:
    """Wraps a fastMRI dataloader + pretrained VarNet as a line-acquisition env.

    Args:
        loader: Yields batches with `kspace` `[B, C, H, W, 2]` and `target`
            `[B, 1, H, W]`, e.g. from `activemri.datasets.fastmri_data
            .build_fastmri_dataloaders` (same batch format `Sequential_Trainer`
            consumes).
        resolution: `[H, W]`.
        budget: Number of columns to acquire per episode, beyond the
            preselected center region.
        preselect_num: Center columns preselected before the policy acts
            (mirrors `SequentialSampling._init_mask`'s line-constrained path).
        recon_checkpoint: `Sequential_Trainer` checkpoint (e.g.
            `checkpoints/mt_varnet_256_1s/best_model.pt`) to initialize the
            reconstructor + sensitivity net from. The reconstructor is frozen
            -- only the DQN policy is trained here.
        reconstructor: Optional frozen module with `VarNet`'s call signature
            (e.g. `Raw2InsightsRecon`) used instead of the built-in VarNet.
            `recon_checkpoint` is then only used for the sensitivity net that
            builds the policy's observation.
        num_cascades, chans, pools: Must match `recon_checkpoint`'s VarNet
            config (defaults match `mt_varnet_256_1s`).
        reward_mode: `"dense"` (per-step NMSE improvement, reconstruction in
            the observation) or `"sparse"` (L2SR: reward only at the terminal
            step, zero-filled image in the observation, no per-step
            reconstruction).
        action_space: `"line"` (one column per action) or `"point"` (one 2D
            k-space point per action).
        points_per_step: Actions (columns or points) acquired per `step()`.
            `budget` then counts steps, not actions.
        center_block: Side of the preselected fully-sampled center square
            (`"point"` mode only; `"line"` mode uses `preselect_num`).
        observation: `"kspace"`: acquired coil-combined k-space and the second
            image-or-k-space pair, as `encode_observation` packs them (5 channels).
            `"kspace_aligned"`: the same, with the k-space rolled back to DC-centred
            (see `_row_phase`). `"image"`: Re/Im of `sum_c conj(S_c) * ifft(mask * k_c)`,
            the SENSE-combined zero-filled image, plus the mask (3 channels).
            `"image_kspace_aligned"`: Re/Im of the DC-centred acquired coil-combined k-space,
            Re/Im of that SENSE-combined zero-filled image, plus the mask (5 channels).
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        resolution: List[int],
        budget: int,
        preselect_num: int = 10,
        recon_checkpoint: Optional[str] = None,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        reconstructor: Optional[torch.nn.Module] = None,
        sense_mask_method: str = "pre_network",
        device: str = "cuda",
        reward_mode: str = "dense",
        action_space: str = "line",
        points_per_step: int = 1,
        center_block: int = 16,
        observation: str = "kspace",
    ):
        if observation not in OBSERVATIONS:
            raise ValueError(f"observation must be one of {OBSERVATIONS}, got {observation!r}")
        if reward_mode not in ("dense", "sparse"):
            raise ValueError(f"reward_mode must be 'dense' or 'sparse', got {reward_mode!r}")
        if action_space not in ("line", "point"):
            raise ValueError(f"action_space must be 'line' or 'point', got {action_space!r}")
        self.reward_mode = reward_mode
        self.action_space = action_space
        self.points_per_step = points_per_step
        self.center_block = center_block
        self.observation = observation
        self.loader = loader
        self._iterator = iter(loader)
        self.shape = resolution
        self.budget = budget
        self.preselect_num = preselect_num
        self.sense_mask_method = sense_mask_method
        self.device = torch.device(device)
        self.num_lines = resolution[1]
        self.num_actions = resolution[0] * resolution[1] if action_space == "point" else resolution[1]

        self.sensetivity_net = SensetivityModel_mc(2, 2, 8).to(self.device)
        self.reconstructor = (
            reconstructor if reconstructor is not None
            else VarNet(num_cascades=num_cascades, chans=chans, pools=pools)
        ).to(self.device)
        if recon_checkpoint is not None:
            self._load_reconstructor(recon_checkpoint, load_recon=reconstructor is None)
        self.sensetivity_net.eval()
        self.reconstructor.eval()
        for p in self.sensetivity_net.parameters():
            p.requires_grad_(False)
        for p in self.reconstructor.parameters():
            p.requires_grad_(False)

        self._state: Dict[str, Any] = {}
        self._episode_start: Optional[Dict[str, Any]] = None

    def _load_reconstructor(self, checkpoint_path: str, load_recon: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        recon_sd = {
            k[len("reconstructor."):]: v for k, v in state_dict.items() if k.startswith("reconstructor.")
        }
        sense_sd = {
            k[len("sensetivity_net."):]: v for k, v in state_dict.items() if k.startswith("sensetivity_net.")
        }
        if load_recon:
            self.reconstructor.load_state_dict(recon_sd)
        self.sensetivity_net.load_state_dict(sense_sd)

    def load_reconstructor_state(self, recon_state_dict: dict, sense_state_dict: dict):
        """Hot-swaps the frozen reconstructor/sensitivity weights in place.

        For `JointTrainer`: after each EMA update of an external online
        reconstructor, push the new EMA weights in here so subsequent
        `step()` calls compute reward/observations against them -- without
        rebuilding the env (dataloader iterator, etc.) from scratch.
        """
        self.reconstructor.load_state_dict(recon_state_dict)
        self.sensetivity_net.load_state_dict(sense_state_dict)

    @property
    def state(self) -> Dict[str, Any]:
        """Read-only access to the current episode's tensors (`full_kspace`,
        `sense_maps`, `mask`, `target`, ...) -- for `JointTrainer` to build a
        supervised training batch for the online reconstructor out of
        whatever mask the policy just selected, without duplicating dataset
        iteration."""
        return self._state

    def rewind(self):
        """Restarts iteration at the first batch, so repeated evaluations score the same slices."""
        self._iterator = iter(self.loader)

    def _next_batch(self) -> Dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)

    @property
    def obs_channels(self) -> int:
        return 3 if self.observation == "image" else 5

    @property
    def target_fraction(self) -> float:
        """Fraction of k-space acquired once an episode ends."""
        h, w = self.shape
        if self.action_space == "point":
            return (self.center_block**2 + self.budget * self.points_per_step) / (h * w)
        return (self.preselect_num + self.budget * self.points_per_step) / w

    def _init_mask(self, batch_size: int) -> torch.Tensor:
        h, w = self.shape
        if self.action_space == "point":
            return center_block_mask(batch_size, self.shape, self.center_block, self.device)
        mask = torch.zeros(batch_size, 1, h, w, device=self.device)
        n = self.preselect_num // 2
        mask[:, :, :, :n] = 1
        mask[:, :, :, -n:] = 1
        return torch.fft.fftshift(mask, dim=(-1, -2))

    def _reconstruct(self, mask: torch.Tensor) -> torch.Tensor:
        """Coil-combined complex reconstruction for the current mask, `[B, 1, H, W]`."""
        masked_kspace = mask.unsqueeze(-1) * self._state["full_kspace"]
        masked_kspace_complex = torch.view_as_complex(masked_kspace.contiguous())
        with torch.no_grad():
            return self.reconstructor(masked_kspace_complex, self._state["sense_maps"], mask)

    def _row_phase(self) -> torch.Tensor:
        """`[1, H, 1]` of `(-1)^row`: the phase the sensitivity net's maps carry along the rows.

        The maps' extra `(-1)^row` factor is harmless to the reconstruction, which only keeps
        magnitudes, but it modulates the sense-combined image, which rolls its k-space by H/2
        rows: DC lands on row 0 instead of the DC-centred row the mask uses.
        """
        rows = torch.arange(self.shape[0], device=self.device)
        return (1.0 - 2.0 * (rows % 2)).view(1, -1, 1)

    def _check_row_phase(self, single_coil_k: torch.Tensor):
        """Fails loudly if rolling the combined k-space by H/2 rows does not move the DC energy to the centre."""
        h, w = self.shape
        block = (slice(h // 2 - 8, h // 2 + 8), slice(w // 2 - 8, w // 2 + 8))
        raw = single_coil_k.abs()[(slice(None),) + block].sum()
        rolled = torch.roll(single_coil_k, shifts=h // 2, dims=-2).abs()[(slice(None),) + block].sum()
        if rolled < ROW_PHASE_MIN_GAIN * raw:
            raise RuntimeError(
                "The sense maps do not carry the (-1)^row phase this observation assumes "
                f"(central-block |k| share, rolled / raw = {float(rolled / raw):.2f})."
            )

    def _zero_filled_image(self, mask: torch.Tensor) -> torch.Tensor:
        """SENSE-combined zero-filled image of the acquired multicoil data, complex `[B, H, W]`."""
        masked = torch.view_as_complex((mask.unsqueeze(-1) * self._state["full_kspace"]).contiguous())
        coil_images = ifft_2d_img(masked)
        return (coil_images * self._state["sense_maps"].conj()).sum(dim=1) * self._row_phase()

    def _encode_obs(self, mask: torch.Tensor, recon: Optional[torch.Tensor]) -> torch.Tensor:
        """Packs (acquired k-space, current-recon k-space, mask) into `[B, 5, H, W]`.

        In sparse mode there is no current reconstruction, so the second pair
        of channels is the zero-filled image (re/im) instead.
        """
        if self.observation == "image":
            image = torch.view_as_real(self._zero_filled_image(mask)).permute(0, 3, 1, 2)
            return torch.cat([image, mask], dim=1)
        if self.observation == "image_kspace_aligned":
            acquired = torch.view_as_real(mask.squeeze(1) * self._state["single_coil_k"]).permute(0, 3, 1, 2)
            image = torch.view_as_real(self._zero_filled_image(mask)).permute(0, 3, 1, 2)
            return torch.cat([acquired, image, mask], dim=1)
        single_coil_k = self._state["single_coil_k"]
        acquired = mask.squeeze(1) * single_coil_k
        second = ifft_2d_img(acquired) if self.reward_mode == "sparse" else fft_2d_img(recon.squeeze(1))
        return encode_observation(mask, single_coil_k, second)

    def _nmse_to_target(self, recon: torch.Tensor) -> torch.Tensor:
        """NMSE of the reconstruction's magnitude against the (real) magnitude target.

        The reconstruction is complex; comparing it to the real target directly
        would make the score depend on the (arbitrary) image phase.
        """
        target = self._state["target"]
        return _nmse(recon.squeeze(1).abs(), target.squeeze(1) if target.dim() == 4 else target)

    def _load_episode_start(self) -> Dict[str, Any]:
        """Draws the next batch and computes everything an episode on it starts from."""
        batch = self._next_batch()
        full_kspace = batch["kspace"].to(self.device)  # [B, C, H, W, 2]
        target = batch["target"].to(self.device)
        full_kspace_complex = torch.view_as_complex(full_kspace.contiguous())

        with torch.no_grad():
            sense_maps, _ = self.sensetivity_net(full_kspace_complex, mask_method=self.sense_mask_method)
            single_coil_k = coil_combined_kspace(full_kspace_complex, sense_maps)
            if self.observation != "kspace":
                self._check_row_phase(single_coil_k)
            if self.observation in ("kspace_aligned", "image_kspace_aligned"):
                single_coil_k = torch.roll(single_coil_k, shifts=self.shape[0] // 2, dims=-2)

        mask = self._init_mask(full_kspace.shape[0])
        self._state = {
            "full_kspace": full_kspace,
            "sense_maps": sense_maps,
            "single_coil_k": single_coil_k,
            "target": target,
        }
        recon = self._reconstruct(mask)
        return {
            **self._state,
            "mask": mask,
            "recon": recon,
            "nmse": self._nmse_to_target(recon),
            "meta": {"fname": batch.get("fname"), "slice_id": batch.get("slice_id")},
        }

    def reset(self, same_batch: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Starts an episode on the next batch, or with `same_batch` on the previous episode's batch.

        Replaying a batch lets a trainer compare several policy draws on the same slices
        (e.g. a leave-one-out baseline), without re-running the sensitivity net or the
        initial reconstruction.
        """
        if not same_batch or self._episode_start is None:
            self._episode_start = self._load_episode_start()
        start = self._episode_start
        self._state = {key: value for key, value in start.items() if key != "meta"}
        self._state["steps_taken"] = 0

        obs = self._encode_obs(start["mask"], start["recon"])
        return obs, start["meta"]

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, List[bool], Dict[str, Any]]:
        """Acquires `actions` and advances one step.

        Args:
            actions: One action per batch element (`List[int]` / `[B]`) or
                `points_per_step` actions per element (`[B, K]` long tensor).
                A line action is a column index; a point action is a flat
                `row * W + col` index.
        """
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        if actions.dim() == 1:
            actions = actions[:, None]
        mask = self._state["mask"].clone()
        if self.action_space == "point":
            mask.view(mask.shape[0], -1).scatter_(1, actions, 1.0)
        else:
            mask[:, 0].scatter_(2, actions[:, None, :].expand(-1, mask.shape[2], -1), 1.0)
        self._state["mask"] = mask
        self._state["steps_taken"] += 1

        done = self._state["steps_taken"] >= self.budget
        dones = [done] * mask.shape[0]

        if self.reward_mode == "sparse" and not done:
            obs = self._encode_obs(mask, None)
            return obs, torch.zeros(mask.shape[0], device=self.device), dones, {}

        recon = self._reconstruct(mask)
        new_nmse = self._nmse_to_target(recon)
        reward = self._state["nmse"] - new_nmse  # positive = improvement
        if self.reward_mode == "dense":
            self._state["nmse"] = new_nmse
        self._state["recon"] = recon

        obs = self._encode_obs(mask, recon)
        meta = {"current_score": {"nmse": new_nmse.detach().cpu().numpy()}}
        return obs, reward.detach(), dones, meta
