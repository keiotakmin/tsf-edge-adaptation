"""Stage-0b contenders: optimizers DESIGNED for non-stationary online prediction.

Stage 0 (stage0_optimizers.py) asked whether existing *general-purpose* memory-light /
LR-free optimizers already occupy the niche the paper's recipe leaves open. They do not
(AdaFactor is the only survivor; every LR-free method collapses under drift). Stage 0b asks
the next question: a 30-year line of work has built optimizers specifically for online
prediction under non-stationarity -- in online convex optimization (Online Newton Step /
discounted RLS), in step-size meta-learning (IDBD), and in streaming RL / continual learning
(ObGD, UPGD). None has ever been evaluated on a deep time-series forecasting backbone.

Every class here is a plain torch.optim.Optimizer taking (params, lr=...), so it drops into
`make_online_optimizer` and inherits the paper's protocol unchanged: fair warmup (C1),
non-overlapping stride=H (C2), and per-cell online-LR rehearsal on the held-out pre-drift
slice (C3). `lr` is always mapped to the method's base/initial step so the shared rehearsal
grid applies; each method's own meta-parameters are pinned at published defaults and NOT
swept (a stated limitation -- sweeping them would re-introduce the C3 unfairness in reverse).

State cost per adapted parameter (what the frontier axis measures):
    obgd      0x   -- no persistent state at lamda=0
    dons      1x   -- discounted second moment
    upgd      1x   -- utility trace
    adaptive_obgd  1x   -- second moment of delta*grad
    idbd      2x   -- log-step-size beta + trace h  (same as Adam: NOT memory-light)
    autostep  3x   -- step alpha + trace h + meta-normaliser v

Optimizers that need the scalar loss set `needs_loss = True`; stream_eval then calls
step(loss=<0-dim tensor>). All reductions stay on-device (no .item()) so the adapt-time
measurement is not polluted by host syncs.
"""
from __future__ import annotations

import torch


class IDBD(torch.optim.Optimizer):
    """Incremental Delta-Bar-Delta (Sutton, AAAI 1992) -- the original step-size
    meta-learning rule, built for online prediction on DRIFTING problems.

    Sutton's LMS form (delta = y - w.x, per-parameter step alpha_i = exp(beta_i)):
        beta_i <- beta_i + theta * delta * x_i * h_i
        w_i    <- w_i + alpha_i * delta * x_i
        h_i    <- h_i * [1 - alpha_i * x_i^2]_+ + alpha_i * delta * x_i

    Generalised here to an arbitrary differentiable loss by substituting the LMS quantities
    with their gradient counterparts: (delta * x_i) -> -g_i, and the diagonal Hessian
    x_i^2 -> the empirical-Fisher proxy g_i^2 (the standard practical approximation; the
    exact diagonal Hessian would need a second backward pass, which an edge update budget
    cannot afford). This gives

        beta_i <- clip(beta_i - theta * g_i * h_i,  log(alpha_min), log(alpha_max))
        alpha_i = exp(beta_i)
        w_i    <- w_i - alpha_i * g_i
        h_i    <- h_i * [1 - alpha_i * g_i^2]_+ - alpha_i * g_i

    `lr` = the INITIAL step alpha_0 (beta_0 = log lr), so the shared LR rehearsal grid
    applies and the LR-robustness band is directly comparable with SGD/Adam. The beta clip
    is ours, not Sutton's: without it the meta-update diverges to alpha = inf on the first
    large-gradient window (verified in test_online_optimizers.py).

    State: beta + h = 2x params -- the same footprint as Adam. IDBD buys LR-freedom, not
    memory; that is exactly the trade-off the frontier figure should show.

    MEASURED LIMITATION (2026-08-07). The meta-update has magnitude ~ theta * alpha * g^2,
    because the substitution above folds the residual into BOTH factors: g_i = delta * df/dw_i,
    so g_i^2 = delta^2 (df/dw_i)^2 shrinks as the model fits, whereas Sutton's x_i^2 does not.
    At the gradient scale of these TSF backbones (|g| ~ 1e-3) that is ~1e-11 per step, so over
    the ~1e3 online updates of a cell beta never leaves beta_0 and IDBD DEGENERATES TO SGD at
    lr = alpha_0: measured identical to torch SGD on 296/342 grid points at L=96 and 283/327 at
    L=192, differing only at the largest alpha_0. Sutton's own testbed has |g| ~ O(1) and so
    cannot expose this -- see the scale test in test_online_optimizers.py. `Autostep` below is
    the published fix and is the variant to read for the step-size-meta-learning question.
    """

    def __init__(self, params, lr=1e-3, theta=1e-2, alpha_min=1e-8, alpha_max=1.0):
        if lr <= 0:
            raise ValueError("IDBD needs lr > 0 (it is the initial step size alpha_0)")
        super().__init__(params, dict(lr=lr, theta=theta, alpha_min=alpha_min,
                                      alpha_max=alpha_max))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            theta = group["theta"]
            lo = float(torch.log(torch.tensor(group["alpha_min"])))
            hi = float(torch.log(torch.tensor(group["alpha_max"])))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["beta"] = torch.full_like(p, float(torch.log(torch.tensor(group["lr"]))))
                    st["h"] = torch.zeros_like(p)
                beta, h = st["beta"], st["h"]
                beta.sub_(g * h, alpha=theta).clamp_(lo, hi)
                alpha = beta.exp()
                p.sub_(alpha * g)
                # h <- h * [1 - alpha g^2]_+ - alpha g
                h.mul_((1 - alpha * g * g).clamp_(min=0)).sub_(alpha * g)


class Autostep(torch.optim.Optimizer):
    """Autostep (Mahmood, Sutton, Degris & Pilarski, AISTATS 2012) -- the published,
    scale-robust successor to IDBD, and the variant that actually answers "does step-size
    meta-learning help?" on gradients this small.

    Autostep changes two things relative to IDBD, and both matter here:
      (1) the meta-update is NORMALISED by a running maximum v of the meta-gradient, so
          alpha <- alpha * exp(mu * m / v) moves by an O(1) amount no matter what absolute
          scale m sits at -- this is exactly what rescues the degeneracy documented in IDBD;
      (2) the effective step is normalised so that sum_i alpha_i c_i <= 1, which bounds the
          update the way ObGD's overshoot cap does, but per-parameter.

        m_i = -g_i * h_i                                        (meta-gradient)
        v_i <- max(|m_i|, v_i + (alpha_i c_i / tau)(|m_i| - v_i))
        alpha_i <- alpha_i * exp(mu * m_i / v_i)                 (v_i > 0)
        alpha <- alpha / max(sum_i alpha_i c_i, 1)
        w_i <- w_i - alpha_i g_i
        h_i <- h_i (1 - alpha_i c_i)_+ - alpha_i g_i

    CURVATURE. c_i is the diagonal GAUSS-NEWTON term (df/dw_i)^2 = g_i^2 / delta^2, NOT the
    empirical-Fisher proxy g_i^2 that IDBD above uses. This is the faithful generalisation, not
    a deviation: for a linear model g_i = delta * x_i, so c_i = x_i^2 EXACTLY, recovering the
    LMS algorithm Sutton and Mahmood et al. actually published. The g^2 form silently folds the
    residual in twice and collapses as the model fits -- measured on the scale sweep in
    test_online_optimizers.py, where IDBD and a g^2-Autostep both lose all step differentiation
    below |g| ~ 1e-1 and end up WORSE than the best fixed step, while this form is invariant to
    4 decimal places across four decades of gradient scale. delta^2 comes from the loss; a constant
    convention factor (MSE vs 1/2 delta^2) is absorbed by alpha's adaptation, since alpha and c
    enter the normaliser and the h decay only as the product alpha*c -- it shifts the transient,
    not the fixed point.

    mu = 1e-2, tau = 1e4 are the paper's values and are pinned. `lr` = alpha_0. State:
    alpha + h + v = 3x params -- more than Adam, the honest cost of tuning-free per-parameter
    steps.
    """

    needs_loss = True

    def __init__(self, params, lr=1e-3, mu=1e-2, tau=1e4, eps=1e-12, alpha_max=1.0):
        super().__init__(params, dict(lr=lr, mu=mu, tau=tau, eps=eps, alpha_max=alpha_max))

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):
        if loss is None:
            raise ValueError("Autostep needs the scalar loss: call step(loss=<0-dim tensor>)")
        for group in self.param_groups:
            mu, tau, eps = group["mu"], group["tau"], group["eps"]
            d2 = loss.clamp(min=eps)                       # residual scale for the GN curvature
            entries = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["alpha"] = torch.full_like(p, float(group["lr"]))
                    st["h"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                alpha, h, v = st["alpha"], st["h"], st["v"]
                c = g * g / d2                             # diagonal Gauss-Newton curvature
                m = -g * h                                 # meta-gradient
                torch.maximum(m.abs(), v + (alpha * c / tau) * (m.abs() - v), out=v)
                alpha.mul_((mu * m / v.clamp(min=eps)).exp_()).clamp_(eps, group["alpha_max"])
                entries.append((p, g, alpha, h, c))
            if not entries:
                continue
            # effective-step normalisation: sum_i alpha_i c_i <= 1
            tot = torch.stack([(a * c).sum() for _, _, a, _, c in entries]).sum()
            scale = 1.0 / tot.clamp(min=1.0)
            for p, g, alpha, h, c in entries:
                a = alpha * scale
                p.sub_(a * g)
                h.mul_((1 - a * c).clamp_(min=0)).sub_(a * g)


class ObGD(torch.optim.Optimizer):
    """Overshooting-bounded Gradient Descent (Elsayed, Vasan & Mahmood 2024, "Streaming Deep
    RL Finally Works") -- supervised port. Designed for batch-size-1 streaming updates, which
    is exactly the online-TSF adaptation step.

    The reference RL implementation (github.com/mohmdelsayed/streaming-drl) updates
    w <- w - step * delta * e with an eligibility trace e of the VALUE gradient, and caps

        M = |delta|_bar * kappa * lr * ||e||_1 ,   step = lr / M  if M > 1 else lr

    so a single sample can never move the prediction past its target. In supervised
    regression the error factor already sits inside the loss gradient (g = delta * grad_f),
    so e = g / delta and the same bound reads

        M = kappa * lr * ||g||_1 * max(d, 1) / d ,   d = sqrt(loss)     (scalar error scale)
        w <- w - min(lr, lr / M) * g

    i.e. the L1 norm of the update is capped at ~1/kappa whenever the bound binds. With MSE
    this is closely related to a capped Polyak step (lr_eff ~ loss / ||g||^2); the difference
    is the L1 (not L2) norm and the kappa margin. NOTE the consequence for the LR grid: once
    the cap binds, `lr` cancels out of the update entirely, so a flat benefit-vs-LR curve is
    the EXPECTED signature, not a bug -- that flatness is the measurement of interest.

    lamda = 0 (default) => no trace across windows => ZERO persistent state, the only
    contender at SGD's memory footprint. lamda > 0 recovers the RL trace (= momentum across
    adjacent adaptation windows, decayed by gamma * lamda) at 1x state.
    """

    needs_loss = True

    def __init__(self, params, lr=1.0, kappa=2.0, lamda=0.0, gamma=0.99, eps=1e-8):
        super().__init__(params, dict(lr=lr, kappa=kappa, lamda=lamda, gamma=gamma, eps=eps))

    def _trace(self, group, p):
        """Eligibility trace, or the raw gradient when lamda == 0 (no state allocated)."""
        if group["lamda"] == 0.0:
            return p.grad
        st = self.state[p]
        if "e" not in st:
            st["e"] = torch.zeros_like(p)
        return st["e"].mul_(group["gamma"] * group["lamda"]).add_(p.grad)

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):
        if loss is None:
            raise ValueError("ObGD needs the scalar loss: call step(loss=<0-dim tensor>)")
        for group in self.param_groups:
            eps, lr, kappa = group["eps"], group["lr"], group["kappa"]
            traces = [(p, self._trace(group, p)) for p in group["params"] if p.grad is not None]
            if not traces:
                continue
            g_l1 = torch.stack([e.abs().sum() for _, e in traces]).sum()
            d = loss.clamp(min=eps).sqrt()                 # scalar error scale
            m = kappa * lr * g_l1 * d.clamp(min=1.0) / d
            step = lr / m.clamp(min=1.0)                   # 0-dim tensor: no host sync
            for p, e in traces:
                p.sub_(e * step)


class AdaptiveObGD(ObGD):
    """ObGD with the paper's Adam-style second moment on delta*e (1x state).

        v <- beta2 * v + (1-beta2) * (g)^2 ;  vhat = v / (1 - beta2^t)
        M uses ||e / sqrt(vhat)||_1 ;  w <- w - step * g / sqrt(vhat)

    Same overshoot cap, but measured in the preconditioned geometry -- the variant the
    authors found necessary on harder streaming control tasks.
    """

    def __init__(self, params, lr=1.0, kappa=2.0, lamda=0.0, gamma=0.99, beta2=0.999,
                 eps=1e-8):
        torch.optim.Optimizer.__init__(
            self, params, dict(lr=lr, kappa=kappa, lamda=lamda, gamma=gamma, beta2=beta2,
                               eps=eps))
        self._t = 0

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):
        if loss is None:
            raise ValueError("AdaptiveObGD needs the scalar loss: step(loss=<0-dim tensor>)")
        self._t += 1
        for group in self.param_groups:
            eps, lr, kappa, beta2 = group["eps"], group["lr"], group["kappa"], group["beta2"]
            bc = 1 - beta2 ** self._t
            entries = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = self._trace(group, p)
                st = self.state[p]
                if "v" not in st:
                    st["v"] = torch.zeros_like(p)
                st["v"].mul_(beta2).addcmul_(e, e, value=1 - beta2)
                denom = (st["v"] / bc).sqrt_().add_(eps)
                entries.append((p, e, denom))
            if not entries:
                continue
            g_l1 = torch.stack([(e / dn).abs().sum() for _, e, dn in entries]).sum()
            d = loss.clamp(min=eps).sqrt()
            m = kappa * lr * g_l1 * d.clamp(min=1.0) / d
            step = lr / m.clamp(min=1.0)
            for p, e, dn in entries:
                p.sub_(step * e / dn)


class ObSign(torch.optim.Optimizer):
    """signSGD with a RELATIVE step guard -- Stage-0c, designed from the Stage-0b frontier.

    What the 216-cell grid says (mean benefit at each FIXED global rate):
        signSGD  3e-5:+13.0  1e-4:+14.7  3e-4:+11.4  1e-3:-7.9  3e-3:-39.4  1e-2:-84.4
        AdaFactor                        1e-3:+14.0  3e-3:+14.1  1e-2:+8.6   (never negative)
        ObGD                             1e-3:+9.1   3e-3:+9.6   1e-2..1e-1:+9.8 (flat)
    So signSGD already owns the best zero-state quality AND its good constant transfers (a
    single 1e-4 across all 216 cells beats every per-cell-rehearsed method except none). Its
    defect is one-sided: too small costs 1.7 pt, too large costs 22.6 pt and then collapses.
    ObGD removes the collapse but only by capping the update in ABSOLUTE terms, which pins its
    plateau at +9.8. AdaFactor is the one method that degrades gracefully, and the mechanism it
    has and the others lack is an update cap RELATIVE TO THE PARAMETER SCALE.

    So: keep signSGD's direction (scale-free, and the reason it beats SGD by 1.5 pt at oracle),
    and bound each tensor's step by a fraction of that tensor's own RMS:

        alpha_i = min(lr, tau * RMS(p_i)),    p_i <- p_i - alpha_i * sign(g_i)

    Zero persistent state (RMS is recomputed each step). A strict SUPERSET of signSGD: below
    the knee lr < tau*RMS(p) the update is identical, so its oracle can never be worse; above
    it the rate cancels and the curve plateaus instead of falling off the cliff. tau is
    dimensionless -- "the largest fraction of a layer's own scale one update may move it" --
    and lr stays the single swept hyperparameter, so the comparison against signSGD isolates
    exactly one change.

    MEASURED (2026-08-07, 36 cells at L=96/H=24). The mechanism works and the level did not:
    with tau = 1e-2 the curve is signSGD up to 1e-4 (+13.8) and then FLAT at +2.7 instead of
    collapsing to -36.3, so the cliff is gone (mis1x 17.2 -> 6.5) but the plateau sits far below
    signSGD's peak. The guard was simply pinned 5x too high -- tau*RMS(p) = 1e-2 * 0.05 = 5e-4
    against signSGD's optimum of 1e-4. Since the plateau level of ObSign(tau) is by construction
    exactly RelSign(tau) (verified in the test suite), the already-collected RelSign curve gives
    it for free: +13.1 at tau=1e-3, +12.6 at 3e-3, +2.7 at 1e-2. tau is therefore SWEPT as three
    arms (obsign / obsign_t3e3 / obsign_t1e3) rather than asserted, and its sensitivity is
    reported -- there is no published default to inherit, unlike ObGD's kappa or UPGD's beta.

    IMPLEMENTATION (`foreach`, added 2026-08-31). The rule above is written per parameter
    tensor, which at batch 1 costs about eight tiny kernel launches per tensor -- 31 of them
    for the PatchTST backbone. torch's own optimizers default to the multi-tensor (`foreach`)
    path and touch every tensor in a handful of launches, so a per-parameter loop measured
    against them reports a difference between two IMPLEMENTATIONS as if it were a difference
    between two update rules. `foreach=True` runs the identical arithmetic through
    `torch._foreach_*`.

    The default stays False so the update is bit-for-bit what the 216-cell grid was computed
    with. The two paths differ only in how RMS is reduced -- `p.pow(2).mean().sqrt()` against
    `||p||_2 / sqrt(numel)` -- the same quantity up to floating-point reassociation, and it
    reaches the parameters at all only when `tau*RMS < lr`; above the knee alpha is exactly
    `lr` and the paths are bit-identical. test_online_optimizers.py asserts the agreement.
    """

    def __init__(self, params, lr=1e-3, tau=1e-2, rms_min=1e-3, foreach=False):
        super().__init__(params, dict(lr=lr, tau=tau, rms_min=rms_min))
        self.foreach = foreach
        self._inv_sqrt_n = {}

    def _inv_sqrt_numel(self, ps):
        """1/sqrt(numel) per tensor as 0-dim tensors, cached: it is a property of the model,
        not of the step, and rebuilding it every update would undo the point of `foreach`."""
        key = tuple(p.numel() for p in ps)
        cached = self._inv_sqrt_n.get(key)
        if cached is None or cached[0].device != ps[0].device:
            cached = [torch.full((), n ** -0.5, device=p.device, dtype=p.dtype)
                      for p, n in zip(ps, key)]
            self._inv_sqrt_n[key] = cached
        return cached

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, tau, rms_min = group["lr"], group["tau"], group["rms_min"]
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            if not self.foreach:
                for p in ps:
                    rms = p.pow(2).mean().sqrt().clamp(min=rms_min)
                    p.sub_(p.grad.sign() * (tau * rms).clamp(max=lr))
                continue
            alpha = torch._foreach_norm(ps)                       # ||p||_2 per tensor
            torch._foreach_mul_(alpha, self._inv_sqrt_numel(ps))  # -> RMS(p)
            torch._foreach_clamp_min_(alpha, rms_min)
            torch._foreach_mul_(alpha, tau)
            torch._foreach_clamp_max_(alpha, lr)                  # alpha = min(lr, tau*RMS)
            upd = torch._foreach_sign([p.grad for p in ps])
            torch._foreach_mul_(upd, alpha)
            torch._foreach_sub_(ps, upd)


class RelSign(torch.optim.Optimizer):
    """The same idea with the absolute rate REMOVED: every update moves a tensor by a fixed
    fraction of its own RMS, so the single hyperparameter is dimensionless.

        p_i <- p_i - tau * RMS(p_i) * sign(g_i)

    ObSign asks "does a relative guard remove signSGD's cliff?"; RelSign asks the sharper
    question "is the step better parameterised as a FRACTION OF THE WEIGHT SCALE than as an
    absolute rate?" -- i.e. is the good tau more transferable across dataset x backbone x
    horizon than the good lr. tau is swept on the shared grid, so both get the same search
    width as every other contender. Zero persistent state; this is signSGD with a per-tensor,
    weight-scale-adaptive rate (the sign-geometry analogue of a LARS trust ratio).
    """

    def __init__(self, params, lr=1e-3, rms_min=1e-3, foreach=False):
        super().__init__(params, dict(lr=lr, rms_min=rms_min))   # lr IS tau here
        self.foreach = foreach                  # see ObSign for why this is off by default
        self._inv_sqrt_n = {}

    _inv_sqrt_numel = ObSign._inv_sqrt_numel

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            tau, rms_min = group["lr"], group["rms_min"]
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            if not self.foreach:
                for p in ps:
                    rms = p.pow(2).mean().sqrt().clamp(min=rms_min)
                    p.sub_(p.grad.sign() * (tau * rms))
                continue
            alpha = torch._foreach_norm(ps)
            torch._foreach_mul_(alpha, self._inv_sqrt_numel(ps))
            torch._foreach_clamp_min_(alpha, rms_min)
            torch._foreach_mul_(alpha, tau)                       # no clamp: RelSign has no lr
            upd = torch._foreach_sign([p.grad for p in ps])
            torch._foreach_mul_(upd, alpha)
            torch._foreach_sub_(ps, upd)


class DiscountedONS(torch.optim.Optimizer):
    """Diagonal DISCOUNTED Online Newton Step / RLS with a forgetting factor -- the practical
    instantiation of the oldest "optimizer specialised for online prediction" line there is
    (Hazan, Agarwal & Kale 2007; Anava, Hazan, Mannor & Shamir, COLT 2013 for ARMA; the
    discounted variant tracks time-varying parameters, cf. variable-forgetting-factor RLS).

        s <- gamma * s + g^2                    (discounted curvature, s_0 = 0)
        w <- w - lr * g / (tau + s)             (tau = Tikhonov term, NOT discounted)

    Two things distinguish this from Adam/RMSprop and both are the point:
      * the preconditioner is 1/s, NOT 1/sqrt(s) -- the Newton/least-squares scaling that
        makes the regret bound logarithmic for exp-concave losses (squared loss is one);
      * gamma < 1 is a forgetting factor, so the curvature estimate tracks drift instead of
        averaging over the whole stream (gamma = 1 recovers plain diagonal ONS/AdaGrad-Newton).
    No first moment => 1x state, half of Adam's.

    tau is NOT optional: textbook ONS needs A_0 = tau*I plus a projection onto a bounded
    domain, and dropping both makes the very first update g/g^2 = 1/g blow up (measured:
    -5.8e5 % benefit on appliances/DLinear before this term was added). tau must also stay
    OUTSIDE the recursion, or it decays as gamma^t and the blow-up merely arrives later.
    tau = 1 is pinned so the first update is exactly an SGD step of size lr, and the method
    then interpolates SGD -> Newton as curvature accumulates: while s << tau the update is
    (lr/tau)*g, i.e. lr and tau are exactly redundant, so sweeping lr over 8 decades already
    covers the tau axis except for WHERE the transition sits. That residual sensitivity is
    the same knob NatSR's authors report as their hardest-to-set hyperparameter.
    """

    def __init__(self, params, lr=1e-3, gamma=0.99, eps=1.0):
        super().__init__(params, dict(lr=lr, gamma=gamma, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, gamma, eps = group["lr"], group["gamma"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["v"] = torch.zeros_like(p)
                v = st["v"].mul_(gamma).addcmul_(g, g, value=1.0)
                p.addcdiv_(g, v + eps, value=-lr)


class UPGD(torch.optim.Optimizer):
    """Utility-based Perturbed Gradient Descent, first-order weight-wise with GLOBAL utility
    scaling (Elsayed & Mahmood, ICLR 2024) -- an optimizer for CONTINUAL learning that needs
    neither task boundaries nor a replay buffer, i.e. it fits the online-TSF setting exactly.

    Transcribed from the authors' FirstOrderGlobalUPGD
    (github.com/mohmdelsayed/upgd, core/optim/weight_upgd/first_order.py):

        u      <- beta_u * u + (1 - beta_u) * (-g * w)          # first-order weight utility
        s      =  sigmoid( (u / (1 - beta_u^t)) / max_global(u) )
        w      <- w - lr * (g + sigma * noise) * (1 - s)

    Useful weights (high utility) are shielded from both the gradient and the perturbation,
    which is what protects against forgetting; low-utility weights get perturbed, which is
    what restores plasticity. State = the utility trace = 1x params.

    CAVEAT for this benchmark: the plasticity-loss effect UPGD targets needs a long stream to
    appear. At stride=H the test stream here is O(1e3) updates per cell, so a null result
    here is evidence about THIS regime, not about UPGD in general. beta_utility and sigma are
    pinned at the paper's values and not swept.
    """

    def __init__(self, params, lr=1e-3, beta_utility=0.999, sigma=1e-3, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, beta_utility=beta_utility, sigma=sigma,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        gmax = None
        for group in self.param_groups:
            bu = group["beta_utility"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["u"] = torch.zeros_like(p)
                st["step"] += 1
                st["u"].mul_(bu).add_(-p.grad * p, alpha=1 - bu)
                m = st["u"].max()
                gmax = m if gmax is None else torch.maximum(gmax, m)
        if gmax is None:
            return
        for group in self.param_groups:
            lr, bu, sigma, wd = (group["lr"], group["beta_utility"], group["sigma"],
                                 group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                s = torch.sigmoid((st["u"] / (1 - bu ** st["step"])) / gmax)
                noise = torch.randn_like(p) * sigma
                if wd:
                    p.mul_(1 - lr * wd)
                p.sub_(lr * (p.grad + noise) * (1 - s))
