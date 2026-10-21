"""Anatomy conditioning: a learned vector per anatomy, injected into a network.

The sampler and the reconstructor otherwise see only k-space, and in a mixed
brain/knee run they converge on a single dataset-average pattern. Giving them
the anatomy identity lets each anatomy own part of the network's behaviour, so
the learned mask can differ between them.

Three injection sites are provided:

    FiLM                per-channel scale and shift, for a feature map or vector
    broadcast_channels  extra constant input channels, for a conv stack
    AnatomyPrompt       a spatial prompt map mixed by the features and the anatomy (PromptMR)

FiLM starts as the identity (zero-initialised projection / zero-mean embedding
contributions are learned from scratch), so enabling it does not perturb an
otherwise-tuned model at step 0.
"""
import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_EMBED_DIM = 16


class AnatomyEmbedding(nn.Module):
    """Maps an anatomy index to a dense conditioning vector.

    Args:
        num_classes: Number of anatomies in the run.
        embed_dim: Width of the conditioning vector.
    """

    def __init__(self, num_classes, embed_dim=DEFAULT_EMBED_DIM):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, anatomy_labels, batch_size):
        """
        Args:
            anatomy_labels: Long tensor [N] of anatomy indices, or None.
            batch_size: Batch size to produce when labels are missing.

        Returns:
            Conditioning vectors [N, embed_dim]. With no labels the mean
            embedding is used, so a conditioned model still runs on data whose
            anatomy is unknown (single-anatomy inference, older loaders).
        """
        weight = self.embedding.weight
        if anatomy_labels is None:
            return weight.mean(dim=0, keepdim=True).expand(batch_size, -1)

        labels = anatomy_labels.to(device=weight.device, dtype=torch.long)
        embedding = self.embedding(labels)
        if embedding.shape[0] != batch_size:
            embedding = embedding.expand(batch_size, -1)
        return embedding


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector.

    The projection is zero-initialised so the layer is the identity until the
    conditioning learns to do something useful.
    """

    def __init__(self, embed_dim, num_features):
        super().__init__()

        self.proj = nn.Linear(embed_dim, 2 * num_features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, embedding):
        scale, shift = self.proj(embedding).chunk(2, dim=1)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


class AnatomyPrompt(nn.Module):
    """PromptMR's prompt block, with the anatomy added to the prompt selection.

    `prompt_len` learned prompt maps are mixed by softmax weights computed from the pooled features
    (PromptMR's input-adaptive prompt) plus the anatomy embedding, resized to the features and passed
    through a 3x3 conv. The caller fuses the result into the features. Fed one constant label, it is
    PromptMR's prompt block with the same parameters, so a label-blind control differs only in the label.

    Args:
        chans: Channels of the features the selection is pooled from.
        embed_dim: Width of the anatomy embedding; 0 selects from the features alone, as PromptMR does.
        prompt_chans: Channels of the prompt map.
        prompt_len: Number of prompt maps mixed.
        prompt_size: Side of the stored prompt maps, before resizing to the features.
    """

    def __init__(self, chans, embed_dim, prompt_chans, prompt_len=5, prompt_size=16):
        super().__init__()

        self.prompts = nn.Parameter(torch.rand(prompt_len, prompt_chans, prompt_size, prompt_size))
        self.feature_weights = nn.Linear(chans, prompt_len)
        self.anatomy_weights = nn.Linear(embed_dim, prompt_len, bias=False) if embed_dim > 0 else None
        self.conv = nn.Conv2d(prompt_chans, prompt_chans, kernel_size=3, padding=1, bias=False)

    def forward(self, features, embedding=None):
        logits = self.feature_weights(features.mean(dim=(-2, -1)))
        if self.anatomy_weights is not None:
            logits = logits + self.anatomy_weights(embedding)
        prompt = torch.einsum('bk,kchw->bchw', logits.softmax(dim=1), self.prompts)
        prompt = F.interpolate(prompt, size=features.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(prompt)


def broadcast_channels(embedding, height, width):
    """Expand [N, embed_dim] to [N, embed_dim, H, W] for concatenation onto an image."""
    return embedding[:, :, None, None].expand(-1, -1, height, width)
