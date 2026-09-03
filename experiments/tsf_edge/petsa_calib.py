"""PETSA's calibration modules and loss, ported into this repository's protocol (G1).

WHY THIS EXISTS. The conference paper's `calib` strategy is a SIMPLIFIED calibration point
"inspired by PETSA" -- a per-channel affine input calibration (2C parameters) plus the linear
output head -- and its Limitations section promises a direct comparison with the official
implementation. This module is that comparison's method side.

WHAT IS PORTED, AND WHAT IS NOT. PETSA (Medeiros et al., ICML 2025 PUT workshop; code
github.com/BorealisAI/PETSA @ 87853d888e98311ac94e64be920d17b57143b20c, Apache-2.0) is built
on top of TAFAS and therefore ships two separable things:

  1. a PARAMETERISATION -- gated low-rank input/output calibration modules around a frozen
     forecaster -- and a composite objective (Huber + frequency L1 + patch-wise structure);
  2. an online SCHEDULE -- period-based batch sizing, adaptation on partially revealed
     targets, and post-hoc adjustment of already-emitted predictions.

Only (1) is ported. (2) is a different evaluation protocol, and adopting it would make the
comparison unreadable: adapting on partially revealed targets and then rewriting predictions
is precisely the family of choices this paper's leakage protocol exists to control. Running
PETSA's parameterisation inside our stride=H, leak-free stream with the same warmup and the
same rehearsed learning-rate grid answers the question the Limitations section actually asks
-- "is our simplification of the calibration MODULE costing accuracy?" -- and leaves the
schedule comparison as stated future work.

FIDELITY NOTES (deviations are here, not hidden in the code):
  * batch size is 1 window, because that is this paper's streaming protocol. PETSA's
    `loss_var` term is a KL between softmaxes taken over the BATCH axis, so at batch 1 it is
    identically zero. It is still computed, so the code path stays faithful, but it cannot
    contribute. PETSA's own period-based batching is part (2) above.
  * the original calls `F.softmax(x)` with no `dim` and `F.kl_div(input, target)` with
    probabilities rather than log-probabilities. Both are reproduced exactly (torch's implicit
    softmax dim for a 3-D input is 0), because the published method is what the code does, not
    what the API documents. They are flagged here so a reader is not left guessing.
  * for PatchTST the original applies OUTPUT calibration only (`CALI_MODULE and MODEL.NAME !=
    'PatchTST'` gates the input side) and builds the module with n_var=1. Our PatchTST is
    channel-independent, so the module is applied per channel with the channels folded into
    the batch axis -- the faithful reading of n_var=1, and it keeps the parameter count
    independent of the meter count. DLinear gets both sides with n_var=C, as in the original.

Defaults are the repository's: rank 16, gating_init 0.01, loss alpha 0.1 (the paper's scripts
pass 0.2/0.02; both are exposed). The learning rate is NOT taken from PETSA's 0.005 default --
it is rehearsed on the shared grid like every other arm, which is the C2-fair treatment.

This module is deliberately NOT in sync_repro.sh's SCRIPTS list: the public repro tree mirrors
the conference submission, and this is extension work.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RANK, GATING_INIT, LOSS_ALPHA, HUBER_DELTA = 16, 0.01, 0.1, 0.5


class GCM(nn.Module):
    """Gated calibration module, transcribed from tta/petsa.py::GCM.

        W = A B,   x <- x + (tanh(gating * x) @ W + bias)

    with A (window_len, r), B (r, window_len, n_var), bias (window_len, n_var), gating (n_var).
    B is zero-initialised, so the module is the identity at step 0 -- the adapted model starts
    exactly at the frozen forecaster, which is what makes it comparable with our `calib`.
    """

    def __init__(self, window_len, n_var=1, gating_init=GATING_INIT, low_rank=RANK):
        super().__init__()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.lora_A = nn.Parameter(torch.empty(window_len, low_rank))
        self.lora_B = nn.Parameter(torch.empty(low_rank, window_len, n_var))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):                              # x: (B, window_len, n_var)
        weight = torch.einsum("ik,kjl->ijl", self.lora_A, self.lora_B)
        return x + (torch.einsum("biv,iov->bov", torch.tanh(self.gating * x), weight)
                    + self.bias)


def _corr_loss(pred, gt):
    """tta/petsa.py::CorrCoefLoss -- negative Pearson correlation over the flattened tensors."""
    return -torch.corrcoef(torch.stack([pred.reshape(-1), gt.reshape(-1)], dim=0))[0, 1]


def petsa_loss(pred, gt, alpha=LOSS_ALPHA):
    """The composite objective of tta/petsa.py, term for term and in the same order."""
    loss_feq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(gt, dim=1)).abs().mean()
    loss = F.huber_loss(pred, gt, delta=HUBER_DELTA) + alpha * loss_feq
    coss = _corr_loss(pred, gt)
    # dim=0 reproduces torch's implicit softmax dim for a 3-D input, which is what the original
    # gets by calling F.softmax with no dim. At batch 1 both softmaxes are all-ones and this
    # term is exactly 0 (see FIDELITY NOTES).
    sf_pred = F.softmax(pred - pred.mean(dim=1, keepdim=True), dim=0)
    sf_gt = F.softmax(gt - gt.mean(dim=1, keepdim=True), dim=0)
    loss_var = F.kl_div(sf_pred, sf_gt).mean()         # probs, not log-probs: as published
    loss_mean = F.l1_loss(pred.mean(dim=1, keepdim=True), gt.mean(dim=1, keepdim=True))
    return loss + (coss + loss_var + loss_mean)


class _Wrapped(nn.Module):
    """frozen forecaster + calibration; channel-folding for the channel-independent backbone"""

    def __init__(self, model, backbone, L, H, C, rank=RANK, gating_init=GATING_INIT):
        super().__init__()
        self.model, self.per_channel = model, (backbone == "patchtst")
        nv = 1 if self.per_channel else C
        # input calibration is skipped for PatchTST, exactly as the original gates it
        self.in_cali = None if self.per_channel else GCM(L, nv, gating_init, rank)
        self.out_cali = GCM(H, nv, gating_init, rank)

    def forward(self, x):                              # x: (B, L, C)
        if self.in_cali is not None:
            x = self.in_cali(x)
        y = self.model(x)                              # (B, H, C)
        if not self.per_channel:
            return self.out_cali(y)
        B, H, C = y.shape
        flat = y.permute(0, 2, 1).reshape(B * C, H, 1)
        return self.out_cali(flat).reshape(B, C, H).permute(0, 2, 1)


def make_wrap(rank=RANK, gating_init=GATING_INIT):
    """-> a `wrap` callable for online_eval.stream_eval."""
    def wrap(model, backbone, L, H, C, device):
        w = _Wrapped(model, backbone, L, H, C, rank, gating_init).to(device)
        extra = [p for n, p in w.named_parameters() if not n.startswith("model.")]
        for p in extra:
            p.requires_grad_(True)
        return w, extra
    return wrap
