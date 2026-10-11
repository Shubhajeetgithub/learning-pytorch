"""
INT4 Quantization in PyTorch
=============================
Implements fake (simulated) INT4 quantization for:
  - Weights            (per-channel, symmetric)
  - Activations/Inputs (per-tensor, asymmetric, calibrated via min/max observer)

Because PyTorch has no native int4 dtype/kernels, quantization is simulated via
quantize -> dequantize ("fake quant"). This is exactly how QAT (quantization-
aware training) works in practice (e.g. PyTorch's own FakeQuantize, or
NVIDIA/Intel INT4 QAT flows): the forward pass sees quantization *noise*, and
gradients flow through it via a Straight-Through Estimator (STE).

At the end, `pack_int4` / `unpack_int4` show how to actually store two 4-bit
values per byte for real deployment/memory savings.

Sections:
  1. Bit-width configuration & qparam (scale/zero-point) computation
  2. STE fake-quantize autograd Function
  3. Observers (calibrate activation ranges)
  4. FakeQuantize modules (weight + activation)
  5. QuantLinear: a Linear layer with INT4 weights & INT4 activations
  6. Real INT4 packing/unpacking (2 values per uint8)
  7. End-to-end example: calibrate -> QAT fine-tune -> deploy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# 1. Bit-width configuration & quantization parameters
# ----------------------------------------------------------------------------

@dataclass
class QuantConfig:
    bits: int = 4
    symmetric: bool = True     # symmetric range around 0 (good for weights)
    per_channel: bool = False  # per-output-channel scale (good for weights)
    ch_axis: int = 0           # which dim is "channel" when per_channel=True

    @property
    def qmin(self):
        return -(2 ** (self.bits - 1)) if self.symmetric else 0

    @property
    def qmax(self):
        return 2 ** (self.bits - 1) - 1 if self.symmetric else 2 ** self.bits - 1


def compute_qparams(x: torch.Tensor, cfg: QuantConfig, eps: float = 1e-8):
    """Compute (scale, zero_point) from the min/max of a tensor."""
    if cfg.per_channel:
        dims = [d for d in range(x.dim()) if d != cfg.ch_axis]
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()

    if cfg.symmetric:
        max_abs = torch.max(x_min.abs(), x_max.abs()).clamp_min(eps)
        scale = max_abs / cfg.qmax
        zero_point = torch.zeros_like(scale)
    else:
        x_min = torch.minimum(x_min, torch.zeros_like(x_min))
        x_max = torch.maximum(x_max, torch.zeros_like(x_max))
        scale = ((x_max - x_min) / (cfg.qmax - cfg.qmin)).clamp_min(eps)
        zero_point = cfg.qmin - torch.round(x_min / scale)
        zero_point = zero_point.clamp(cfg.qmin, cfg.qmax)

    return scale, zero_point


# ----------------------------------------------------------------------------
# 2. STE fake-quantize
# ----------------------------------------------------------------------------

class _FakeQuantSTE(torch.autograd.Function):
    """quantize -> dequantize with straight-through gradient.

    Forward:  x_q = clamp(round(x/scale + zp), qmin, qmax); x_dq = (x_q - zp) * scale
    Backward: gradient passes through unchanged for values inside [qmin, qmax]
              range, and is zeroed for values that were clipped (standard STE
              variant, sometimes called "PACT-style" clipped STE).
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_scaled = x / scale + zero_point
        x_q = torch.clamp(torch.round(x_scaled), qmin, qmax)
        x_dq = (x_q - zero_point) * scale

        # mask of values that were NOT clipped, for the backward pass
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        ctx.save_for_backward(mask)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None, None, None, None


def fake_quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, cfg: QuantConfig):
    return _FakeQuantSTE.apply(x, scale, zero_point, cfg.qmin, cfg.qmax)


# ----------------------------------------------------------------------------
# 3. Observers — calibrate activation/input ranges from real data
# ----------------------------------------------------------------------------

class MinMaxObserver(nn.Module):
    """Tracks a running min/max of observed activations (EMA), used to derive
    static scale/zero-point after a calibration pass over some sample data."""

    def __init__(self, cfg: QuantConfig, momentum: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.momentum = momentum
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_min, batch_max = x.min(), x.max()
        if not bool(self.initialized):
            self.running_min.copy_(batch_min)
            self.running_max.copy_(batch_max)
            self.initialized.fill_(True)
        else:
            self.running_min.mul_(1 - self.momentum).add_(self.momentum * batch_min)
            self.running_max.mul_(1 - self.momentum).add_(self.momentum * batch_max)

    def qparams(self):
        eps = 1e-8
        if self.cfg.symmetric:
            max_abs = torch.max(self.running_min.abs(), self.running_max.abs()).clamp_min(eps)
            scale = max_abs / self.cfg.qmax
            zero_point = torch.zeros_like(scale)
        else:
            lo = torch.minimum(self.running_min, torch.zeros_like(self.running_min))
            hi = torch.maximum(self.running_max, torch.zeros_like(self.running_max))
            scale = ((hi - lo) / (self.cfg.qmax - self.cfg.qmin)).clamp_min(eps)
            zero_point = torch.clamp(
                torch.round(self.cfg.qmin - lo / scale), self.cfg.qmin, self.cfg.qmax
            )
        return scale, zero_point


# ----------------------------------------------------------------------------
# 4. FakeQuantize modules
# ----------------------------------------------------------------------------

class WeightFakeQuant(nn.Module):
    """INT4 weight quantizer: per-output-channel, symmetric (standard choice —
    avoids needing a zero-point for weights and matches most HW INT4 kernels)."""

    def __init__(self, bits: int = 4, per_channel: bool = True, ch_axis: int = 0):
        super().__init__()
        self.cfg = QuantConfig(bits=bits, symmetric=True, per_channel=per_channel, ch_axis=ch_axis)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        scale, zero_point = compute_qparams(w.detach(), self.cfg)
        return fake_quantize(w, scale, zero_point, self.cfg)


class ActivationFakeQuant(nn.Module):
    """INT4 activation/input quantizer: per-tensor, asymmetric (activations are
    often non-negative post-ReLU / skewed, so asymmetric range is standard).

    Two modes:
      - calibrating=True : just update the running min/max observer, pass
        input through unchanged (used during a calibration pass).
      - calibrating=False: apply fake quant using the frozen observed qparams
        (used during QAT fine-tuning and at "deployment").
    """

    def __init__(self, bits: int = 4, symmetric: bool = False, momentum: float = 0.1):
        super().__init__()
        self.cfg = QuantConfig(bits=bits, symmetric=symmetric, per_channel=False)
        self.observer = MinMaxObserver(self.cfg, momentum=momentum)
        self.calibrating = True
        self.enabled = True  # allow fully disabling quant (e.g. for fp32 baseline)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.observer.update(x.detach())
            return x
        if not self.enabled:
            return x
        scale, zero_point = self.observer.qparams()
        return fake_quantize(x, scale, zero_point, self.cfg)


# ----------------------------------------------------------------------------
# 5. QuantLinear — INT4 weights + INT4 activations (drop-in for nn.Linear)
# ----------------------------------------------------------------------------

class QuantLinear(nn.Module):
    """Linear layer that fake-quantizes BOTH its input activations and its
    weights to INT4 on every forward pass. Bias is kept in fp32 (standard
    practice — bias has negligible memory cost and quantizing it hurts
    accuracy disproportionately)."""

    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=4, act_bits=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_quant = WeightFakeQuant(bits=weight_bits, per_channel=True, ch_axis=0)
        self.act_quant = ActivationFakeQuant(bits=act_bits, symmetric=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)

    def set_calibrating(self, mode: bool):
        self.act_quant.calibrating = mode


@torch.no_grad()
def calibrate(model: nn.Module, data_loader, num_batches: int = 20, device="cpu"):
    """Run a few batches through the model in 'calibrating' mode so every
    ActivationFakeQuant's observer sees real activation statistics."""
    for m in model.modules():
        if isinstance(m, ActivationFakeQuant):
            m.calibrating = True
    model.eval()
    for i, (x, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        model(x.to(device))
    for m in model.modules():
        if isinstance(m, ActivationFakeQuant):
            m.calibrating = False


# ----------------------------------------------------------------------------
# 6. Real INT4 packing (2 values per byte) — for actual storage/inference
# ----------------------------------------------------------------------------

def pack_int4(x_int: torch.Tensor) -> torch.Tensor:
    """Pack a tensor of int8 values in [-8,7] (or [0,15]) into uint8, two per
    byte. `x_int` must have an even number of elements along its last dim."""
    x_uint = (x_int.to(torch.int16) & 0xF).to(torch.uint8)  # map to 4-bit pattern
    flat = x_uint.flatten()
    assert flat.numel() % 2 == 0, "need an even number of elements to pack"
    lo = flat[0::2]
    hi = flat[1::2]
    packed = (hi << 4) | lo
    return packed


def unpack_int4(packed: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """Inverse of pack_int4. Returns int8 tensor of the original nibble values."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    if signed:
        out = out.to(torch.int8)
        out[out > 7] -= 16  # map [8..15] -> [-8..-1]
    return out


def quantize_to_int4_storage(w: torch.Tensor, cfg: QuantConfig):
    """Full PTQ pipeline for actually storing a weight tensor as packed int4:
    compute qparams -> integer codes -> pack. Returns (packed_bytes, scale, zero_point, shape).
    """
    scale, zero_point = compute_qparams(w, cfg)
    codes = torch.clamp(torch.round(w / scale + zero_point), cfg.qmin, cfg.qmax).to(torch.int8)
    packed = pack_int4(codes)
    return packed, scale, zero_point, w.shape


# ----------------------------------------------------------------------------
# 7. End-to-end example
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = QuantLinear(64, 128, weight_bits=4, act_bits=4)
            self.act = nn.ReLU()
            self.fc2 = QuantLinear(128, 10, weight_bits=4, act_bits=4)

        def forward(self, x):
            x = self.act(self.fc1(x))
            return self.fc2(x)

    model = TinyMLP()

    # ---- fake "calibration" dataset ----
    calib_data = [(torch.randn(32, 64), None) for _ in range(20)]
    calibrate(model, calib_data, num_batches=20)
    print("Calibration done. Example activation qparams (fc1):",
          model.fc1.act_quant.observer.qparams())

    # ---- QAT fine-tune: gradients flow through fake-quant via STE ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(32, 64)
    target = torch.randint(0, 10, (32,))
    for step in range(5):
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, target)
        loss.backward()
        opt.step()
        print(f"step {step}: loss={loss.item():.4f}")

    # ---- "deploy": pack a weight tensor to real int4 for storage ----
    w = model.fc1.linear.weight.detach()
    cfg = QuantConfig(bits=4, symmetric=True, per_channel=True, ch_axis=0)
    packed, scale, zp, shape = quantize_to_int4_storage(w, cfg)
    print(f"\nOriginal weight bytes (fp32): {w.numel() * 4}")
    print(f"Packed int4 bytes:            {packed.numel()}  "
          f"({w.numel() * 4 / packed.numel():.1f}x smaller)")

    # sanity check: unpack and dequantize, compare to fake-quant version
    unpacked = unpack_int4(packed, signed=True).reshape(shape)
    dequant = (unpacked.float() - zp) * scale
    fq = fake_quantize(w, scale, zp, cfg)
    print("Max abs diff (pack/unpack vs fake-quant path):",
          (dequant - fq).abs().max().item())