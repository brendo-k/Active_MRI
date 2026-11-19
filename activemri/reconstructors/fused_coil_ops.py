"""The VarNet cascade's multi-coil path as compiled real-arithmetic kernels with hand-written backward passes.

Forward, for coil k-spaces k, sensitivity maps S, a scale s = 1 / sqrt(H W) and the unnormalised FFT F:

    combine_coils(k, S)                    image = s sum_c conj(S_c) F^H k_c
    expand_and_update(c, S, k, k0, w, m)   k - where(m != 0, k - k0, 0) w - s F (S_c c)

These are the orthonormal-FFT cascade written with the scale folded into the compiled kernels, so an FFT costs no extra
pass over the tensor. Autograd through the same expressions spends a large part of a step on bookkeeping, so
the backward is written out here instead:
- the gradient of a coil operation is another coil operation, one fused kernel, with no zero-filled tensors to index into;
- S and k0 are read by every cascade. Each consumer adds its gradient straight into one running sum (`SharedGradient`),
  inside its own kernel, instead of allocating a full-size tensor for the autograd engine to add.

A backward frees what it saved as it finishes, as autograd does, so these graphs support one backward pass.

The FFTs stay cuFFT calls outside. Complex tensors never enter the compiled graphs (inductor miscomputed a compiled
cascade on complex inputs). The arithmetic matches the eager cascade up to float rounding.
"""
import math

import torch
from torch.autograd import Function

from activemri.utils.compilers import allow_recompiles, use_system_compilers_if_missing

FFT_AXES = (-1, -2)
_COMPILED = {}


def _compiled(fn):
    if fn not in _COMPILED:
        use_system_compilers_if_missing()
        allow_recompiles()
        _COMPILED[fn] = torch.compile(fn, dynamic=False, fullgraph=True)
    return _COMPILED[fn]


def _real(tensor: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(tensor.resolve_conj())


def _fft_scale(tensor: torch.Tensor) -> float:
    return 1.0 / math.sqrt(tensor.shape[-1] * tensor.shape[-2])


def _quarter_turn(z: torch.Tensor) -> torch.Tensor:
    """i z, for a complex tensor stored as [..., 2]."""
    return z.flip(-1) * z.new_tensor([-1.0, 1.0])


def _multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a b, for complex tensors stored as [..., 2]. Written on the stored layout, not on separate real and imaginary
    planes, because a fused kernel then writes the result directly instead of two temporaries and a stack."""
    return a[..., :1] * b + a[..., 1:] * _quarter_turn(b)


def _multiply_conj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """conj(a) b, for complex tensors stored as [..., 2]."""
    return a[..., :1] * b - a[..., 1:] * _quarter_turn(b)


def _combine(x: torch.Tensor, s: torch.Tensor, scale: float) -> torch.Tensor:
    return _multiply_conj(s, x).sum(dim=1, keepdim=True) * scale


def _combine_backward(g: torch.Tensor, x: torch.Tensor, s: torch.Tensor, scale: float, s_sum):
    """Of image = scale sum_c conj(S_c) x_c with gradient g: (g S_c scaled, and s_sum + conj(g) x_c scaled, if tracked)."""
    g = g * scale
    x_grad = _multiply(g, s)
    return x_grad, None if s_sum is None else s_sum + _multiply_conj(g, x)


def _expand(image: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return _multiply(image, s)


def _expand_backward(h: torch.Tensor, image: torch.Tensor, s: torch.Tensor, scale: float, s_sum):
    """Of coils = S_c image with gradient scale h: (sum_c conj(S_c) h_c, and s_sum + h_c conj(image), if tracked)."""
    h = h * scale
    image_grad = _multiply_conj(s, h).sum(dim=1, keepdim=True)
    return image_grad, None if s_sum is None else s_sum + _multiply_conj(image, h)


def _dc_update(current: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, term: torch.Tensor,
               scale: float) -> torch.Tensor:
    keep = (mask != 0).unsqueeze(-1)
    soft_dc = torch.where(keep, current - ref, torch.zeros_like(current)) * weight
    return current - soft_dc - term * scale


def _dc_backward(g: torch.Tensor, current: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, ref_sum):
    """Of next = current - where(mask != 0, current - ref, 0) weight - ... with gradient g: the gradient of `current`,
    ref_sum + the gradient of `ref` if tracked, and the real gradient of `weight`, shape [1]."""
    keep = (mask != 0).unsqueeze(-1)
    dc_grad = torch.where(keep, g * weight, torch.zeros_like(g))
    weight_grad = -(torch.where(keep, current - ref, torch.zeros_like(current)) * g).sum().reshape(1)
    return g - dc_grad, None if ref_sum is None else ref_sum + dc_grad, weight_grad


class _RunningSum:
    """The gradient a shared tensor has collected so far, in real form. It is separate from `SharedGradient` so the
    autograd node holds no reference to its own output, a cycle that would keep the whole tensor alive after each step."""

    value = None


class _CollectGradient(Function):
    """Identity whose backward hands the autograd engine the sum its consumers built. It runs after every consumer,
    because each of them is a dependency of this node, whether or not it returns a gradient for it."""

    @staticmethod
    def forward(ctx, tensor, running_sum):
        ctx.set_materialize_grads(False)
        ctx.running_sum = running_sum
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, _unused):
        value, ctx.running_sum.value = ctx.running_sum.value, None
        return None if value is None else torch.view_as_complex(value), None


class SharedGradient:
    """A complex tensor many fused consumers read. `tensor` is what they use, and if `tracked` they add their gradients
    to `running_sum` instead of returning them."""

    def __init__(self, tensor: torch.Tensor):
        self.tracked = torch.is_grad_enabled() and tensor.requires_grad
        self._sum = _RunningSum()
        self.tensor = _CollectGradient.apply(tensor, self._sum) if self.tracked else tensor
        self.real = _real(self.tensor).detach()

    def running_sum(self):
        """The real view of the sum so far, or None if nothing collects it."""
        if not self.tracked:
            return None
        return torch.zeros_like(self.real) if self._sum.value is None else self._sum.value

    def store(self, value):
        self._sum.value = value


class _CombineCoils(Function):
    @staticmethod
    def forward(ctx, kspace, maps_tensor, maps: SharedGradient):
        coil_images = _real(torch.fft.ifft2(kspace, dim=FFT_AXES, norm='forward'))
        ctx.coil_images, ctx.maps, ctx.scale = coil_images, maps, _fft_scale(kspace)
        return torch.view_as_complex(_compiled(_combine)(coil_images, maps.real, ctx.scale))

    @staticmethod
    def backward(ctx, image_grad):
        coil_grad, maps_sum = _compiled(_combine_backward)(_real(image_grad), ctx.coil_images, ctx.maps.real, ctx.scale,
                                                           ctx.maps.running_sum())
        ctx.maps.store(maps_sum)
        del ctx.coil_images
        return torch.fft.fft2(torch.view_as_complex(coil_grad), dim=FFT_AXES, norm='backward'), None, None


class _ExpandAndUpdate(Function):
    @staticmethod
    def forward(ctx, correction, maps_tensor, current, ref_tensor, weight, mask, maps: SharedGradient, ref: SharedGradient):
        scale = _fft_scale(current)
        correction, current = _real(correction), _real(current)
        coil_images = _compiled(_expand)(correction, maps.real)
        term = torch.fft.fft2(torch.view_as_complex(coil_images), dim=FFT_AXES, norm='backward')
        ctx.correction, ctx.current, ctx.weight, ctx.mask = correction, current, weight, mask
        ctx.maps, ctx.ref, ctx.scale = maps, ref, scale
        return torch.view_as_complex(_compiled(_dc_update)(current, ref.real, mask, weight, _real(term), scale))

    @staticmethod
    def backward(ctx, grad):
        current_grad, ref_sum, weight_grad = _compiled(_dc_backward)(
            _real(grad), ctx.current, ctx.ref.real, ctx.mask, ctx.weight, ctx.ref.running_sum())
        ctx.ref.store(ref_sum)
        coil_grad = _real(torch.fft.ifft2(grad.resolve_conj(), dim=FFT_AXES, norm='forward'))
        correction_grad, maps_sum = _compiled(_expand_backward)(coil_grad, ctx.correction, ctx.maps.real, -ctx.scale,
                                                                ctx.maps.running_sum())
        ctx.maps.store(maps_sum)
        del ctx.correction, ctx.current
        return torch.view_as_complex(correction_grad), None, torch.view_as_complex(current_grad), None, weight_grad, None, None, None


def combine_coils(kspace: torch.Tensor, maps: SharedGradient) -> torch.Tensor:
    """[B, C, H, W] complex uncentered k-space -> [B, 1, H, W] complex image sum_c conj(S_c) IFFT(k_c), orthonormal."""
    return _CombineCoils.apply(kspace, maps.tensor, maps)


def expand_and_update(correction: torch.Tensor, maps: SharedGradient, current: torch.Tensor, ref: SharedGradient,
                      weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """current - where(mask != 0, current - ref, 0) * weight - FFT(S_c correction), orthonormal, for a [B, 1, H, W]
    uncentered complex `correction`, [B, C, H, W] complex k-spaces, a real [B, 1, H, W] mask and a scalar weight of shape [1]."""
    return _ExpandAndUpdate.apply(correction, maps.tensor, current, ref.tensor, weight, mask.detach(), maps, ref)
