import math

import torch
from scipy import integrate
from torch import nn
from torchdiffeq import odeint
from tqdm.auto import tqdm, trange


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + torch.randn_like(x, device="cpu").to(x.device) * sigma_up
    return x


@torch.no_grad()
def sample_heun(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0
):
    """Ancestral sampling with DPM-Solver inspired second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + torch.randn_like(x, device="cpu").to(x.device) * sigma_up
    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        cur_order = min(i + 1, order)
        coeffs = [
            linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
            for j in range(cur_order)
        ]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


@torch.no_grad()
def log_likelihood(
    model, x, sigma_min, sigma_max, extra_args=None, atol=1e-4, rtol=1e-4
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    v = torch.randint_like(x, 2) * 2 - 1
    fevals = 0

    def ode_fn(sigma, x):
        nonlocal fevals
        with torch.enable_grad():
            x = x[0].detach().requires_grad_()
            denoised = model(x, sigma * s_in, **extra_args)
            d = to_d(x, sigma, denoised)
            fevals += 1
            grad = torch.autograd.grad((d * v).sum(), x)[0]
            d_ll = (v * grad).flatten(1).sum(1)
        return d.detach(), d_ll

    x_min = x, x.new_zeros([x.shape[0]])
    t = x.new_tensor([sigma_min, sigma_max])
    sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method="dopri5")
    latent, delta_ll = sol[0][-1], sol[1][-1]
    ll_prior = (
        torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
    )
    return ll_prior + delta_ll, {"fevals": fevals}


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""

    def __init__(
        self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8
    ):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = (
            self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        )
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (
            x - self.model(x, sigma, *args, **self.extra_args, **kwargs)
        ) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        )
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = (
            x
            - self.sigma(s2) * (r2 * h).expm1() * eps
            - self.sigma(s2)
            * (r2 / r1)
            * ((r2 * h).expm1() / (r2 * h) - 1)
            * (eps_r1 - eps)
        )
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        )
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0.0, s_noise=1.0):
        if not t_end > t_start and eta:
            raise ValueError("eta must be 0 for reverse sampling")

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.0

            eps, eps_cache = self.eps(eps_cache, "eps", x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback(
                    {"x": x, "i": i, "t": ts[i], "t_up": t, "denoised": denoised}
                )

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(
                    x, t, t_next_, eps_cache=eps_cache
                )
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(
                    x, t, t_next_, eps_cache=eps_cache
                )
            else:
                x, eps_cache = self.dpm_solver_3_step(
                    x, t, t_next_, eps_cache=eps_cache
                )

            x = x + su * s_noise * torch.randn_like(x)

        return x

    def dpm_solver_adaptive(
        self,
        x,
        t_start,
        t_end,
        order=3,
        rtol=0.05,
        atol=0.0078,
        h_init=0.05,
        pcoeff=0.0,
        icoeff=1.0,
        dcoeff=0.0,
        accept_safety=0.81,
        eta=0.0,
        s_noise=1.0,
    ):
        if order not in {2, 3}:
            raise ValueError("order should be 2 or 3")
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError("eta must be 0 for reverse sampling")
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(
            h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety
        )
        info = {"steps": 0, "nfe": 0, "n_accept": 0, "n_reject": 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = (
                torch.minimum(t_end, s + pid.h)
                if forward
                else torch.maximum(t_end, s + pid.h)
            )
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.0

            eps, eps_cache = self.eps(eps_cache, "eps", x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(
                    x, s, t_, eps_cache=eps_cache
                )
            else:
                x_low, eps_cache = self.dpm_solver_2_step(
                    x, s, t_, r1=1 / 3, eps_cache=eps_cache
                )
                x_high, eps_cache = self.dpm_solver_3_step(
                    x, s, t_, eps_cache=eps_cache
                )
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * torch.randn_like(x_high)
                s = t
                info["n_accept"] += 1
            else:
                info["n_reject"] += 1
            info["nfe"] += order
            info["steps"] += 1

            if self.info_callback is not None:
                self.info_callback(
                    {
                        "x": x,
                        "i": info["steps"] - 1,
                        "t": s,
                        "t_up": s,
                        "denoised": denoised,
                        "error": error,
                        "h": pid.h,
                        **info,
                    }
                )

        return x, info


@torch.no_grad()
def sample_dpm_fast(
    model,
    x,
    sigma_min,
    sigma_max,
    n,
    extra_args=None,
    callback=None,
    disable=None,
    eta=0.0,
    s_noise=1.0,
):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        return dpm_solver.dpm_solver_fast(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            n,
            eta,
            s_noise,
        )


@torch.no_grad()
def sample_dpm_adaptive(
    model,
    x,
    sigma_min,
    sigma_max,
    extra_args=None,
    callback=None,
    disable=None,
    order=3,
    rtol=0.05,
    atol=0.0078,
    h_init=0.05,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    accept_safety=0.81,
    eta=0.0,
    s_noise=1.0,
    return_info=False,
):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        x, info = dpm_solver.dpm_solver_adaptive(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            order,
            rtol,
            atol,
            h_init,
            pcoeff,
            icoeff,
            dcoeff,
            accept_safety,
            eta,
            s_noise,
        )
    if return_info:
        return x, info
    return x
