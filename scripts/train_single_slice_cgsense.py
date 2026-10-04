"""Train sensitivity model + CG-SENSE on a single brain slice (overfit test)."""

import pathlib

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import _path  # noqa: F401
from activemri.reconstructors.CG_Sense import CGSENSEReconstructor
from activemri.reconstructors.SensetivityModel_mc import SensetivityModel_mc
from activemri.utils.image_processing import ifft_2d_img, fft_2d_img, root_sum_of_squares


DATA_ROOT = pathlib.Path("/home/bkadota/data/fastmri/multicoil_brain/T2/train")
IMAGE_SHAPE = (128, 128)
NUM_COILS = 16
CG_ITERS = 10
CG_LAMBDA = 0.0
LR = 1e-3
NUM_EPOCHS = 200
ACCELERATION = 4.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_single_slice():
    """Load one brain slice, preprocess to [1, C, H, W] complex k-space + target."""
    h5_file = sorted(DATA_ROOT.glob("*.h5"))[0]
    with h5py.File(h5_file, "r") as f:
        kspace_np = f["kspace"][8]  # pick middle-ish slice

    kspace = torch.from_numpy(kspace_np)

    # Downsample readout if non-square (brain convention)
    if kspace.shape[-2] > kspace.shape[-1]:
        kspace = kspace[..., ::2, :]
    elif kspace.shape[-1] > kspace.shape[-2]:
        kspace = kspace[..., ::2]

    # Center crop
    h, w = IMAGE_SHAPE
    h_from = (kspace.shape[-2] - h) // 2
    w_from = (kspace.shape[-1] - w) // 2
    kspace = kspace[..., h_from:h_from + h, w_from:w_from + w]

    # Pad/crop coils
    if kspace.shape[0] > NUM_COILS:
        kspace = kspace[:NUM_COILS]
    elif kspace.shape[0] < NUM_COILS:
        pad = torch.zeros(NUM_COILS - kspace.shape[0], *kspace.shape[1:], dtype=kspace.dtype)
        kspace = torch.cat([kspace, pad], dim=0)

    # Flip readout (brain convention)
    kspace = kspace.flip(-2)

    # Normalize by center k-space RSS
    ifft_np = lambda x: np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-1, -2)), axes=(-1, -2), norm="ortho"),
        axes=(-1, -2),
    )
    center_mask = torch.zeros_like(kspace)
    ch, cw = h // 2, w // 2
    center_mask[:, ch - 5:ch + 5, cw - 5:cw + 5] = 1
    center_k = kspace * center_mask
    norm_factor = np.max(np.sqrt(np.sum(np.abs(ifft_np(center_k.numpy())) ** 2, axis=0)))
    kspace = kspace / norm_factor

    # Target: RSS of IFFT
    target = torch.from_numpy(
        np.sqrt(np.sum(np.abs(ifft_np(kspace.numpy())) ** 2, axis=0))
    ).float()

    # Add batch dim
    kspace = kspace.unsqueeze(0)  # [1, C, H, W] complex
    target = target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return kspace, target


def create_random_mask(shape, acceleration, preselect_num=10):
    """Create a random 2D undersampling mask with center pre-selected."""
    B, _, H, W = shape
    budget = int(H * W / acceleration)

    mask = torch.zeros(B, 1, H, W)

    # Pre-select center
    ch, cw = H // 2, W // 2
    n = preselect_num // 2
    mask[:, :, ch - n:ch + n, cw - n:cw + n] = 1

    # Randomly sample remaining budget
    remaining = budget - mask.sum().int().item()
    if remaining > 0:
        for b in range(B):
            available = (mask[b, 0] == 0).nonzero(as_tuple=False)
            idx = torch.randperm(available.shape[0])[:remaining]
            coords = available[idx]
            mask[b, 0, coords[:, 0], coords[:, 1]] = 1

    return mask


def main():
    print(f"Device: {DEVICE}")
    kspace_complex, target = load_single_slice()
    kspace_complex = kspace_complex.to(DEVICE)
    target = target.to(DEVICE)

    print(f"K-space shape: {kspace_complex.shape}")
    print(f"Target shape: {target.shape}")

    # Fixed mask for this experiment
    mask = create_random_mask(
        (1, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        acceleration=ACCELERATION,
        preselect_num=10,
    ).to(DEVICE)

    print(f"Mask density: {mask.sum().item() / mask.numel():.3f}")

    # Models
    sense_net = SensetivityModel_mc(
        in_chans=2, out_chans=2, chans=8,
    ).to(DEVICE)

    cg_recon = CGSENSEReconstructor(num_iters=CG_ITERS, lamda=CG_LAMBDA).to(DEVICE)

    optimizer = torch.optim.Adam(sense_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    losses = []

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        # Estimate sensitivity maps
        sense_maps, _ = sense_net(kspace_complex, mask_method=None)

        # Apply mask to k-space
        masked_kspace = kspace_complex * mask

        # CG-SENSE reconstruction
        recon = cg_recon(masked_kspace, sense_maps, mask)  # [B, 1, H, W] magnitude

        loss = F.l1_loss(recon, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save results
    with torch.no_grad():
        sense_maps, _ = sense_net(kspace_complex, mask_method=None)
        masked_kspace = kspace_complex * mask
        final_recon = cg_recon(masked_kspace, sense_maps, mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(target[0, 0].cpu().numpy(), cmap="gray")
    axes[0].set_title("Target (RSS)")

    axes[1].imshow(final_recon[0, 0].cpu().numpy(), cmap="gray")
    axes[1].set_title("CG-SENSE Recon")

    axes[2].imshow((target[0, 0] - final_recon[0, 0]).abs().cpu().numpy(), cmap="hot")
    axes[2].set_title("Error")

    axes[3].plot(losses)
    axes[3].set_title("Training Loss")
    axes[3].set_xlabel("Epoch")

    for ax in axes[:3]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("single_slice_cgsense_result.png", dpi=150)
    plt.close()
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print("Saved: single_slice_cgsense_result.png")


if __name__ == "__main__":
    main()
