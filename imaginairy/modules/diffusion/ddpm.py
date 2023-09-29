"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci.
"""
import itertools
import logging
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from omegaconf import ListConfig
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from tqdm import tqdm

from imaginairy.modules.attention import CrossAttention
from imaginairy.modules.autoencoder import AutoencoderKL, IdentityFirstStage
from imaginairy.modules.diffusion.util import (
    extract_into_tensor,
    make_beta_schedule,
    noise_like,
)
from imaginairy.modules.distributions import DiagonalGaussianDistribution
from imaginairy.modules.ema import LitEma
from imaginairy.paths import PKG_ROOT
from imaginairy.samplers.kdiff import DPMPP2MSampler
from imaginairy.utils import instantiate_from_config

logger = logging.getLogger(__name__)
__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = []
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype(f"{PKG_ROOT}/data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def disabled_train(self):
    """
    Overwrite model.train with this function to make sure train/eval mode
    does not change anymore.
    """
    return self


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=(),
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        make_it_fit=False,
        ucg_training=None,
        reset_ema=False,
        reset_num_ema_updates=False,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        # print(
        #     f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        # )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        # count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            # print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        self.make_it_fit = make_it_fit
        if reset_ema:
            assert ckpt_path is not None
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )
            if reset_ema:
                assert self.use_ema
                print(
                    "Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint."
                )
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(
                " +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ "
            )
            assert self.use_ema
            self.model_ema.reset_num_updates()

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.ucg_training = ucg_training or {}
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * to_torch(alphas)
                    * (1 - self.alphas_cumprod)
                )
            )
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def init_from_state_dict(self, sd, ignore_keys=(), only_model=False):
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())

        if self.cond_stage_key == "edit":
            # from https://github.com/timothybrooks/instruct-pix2pix/blob/main/stable_diffusion/ldm/models/diffusion/ddpm_edit.py#L203-L221
            input_keys = [
                "model.diffusion_model.input_blocks.0.0.weight",
                "model_ema.diffusion_modelinput_blocks00weight",
            ]

            self_sd = self.state_dict()
            for input_key in input_keys:
                if input_key not in sd or input_key not in self_sd:
                    continue

                input_weight = self_sd[input_key]

                if input_weight.size() != sd[input_key].size():
                    input_weight.zero_()
                    input_weight[:, :4, :, :].copy_(sd[input_key])
                    ignore_keys.append(input_key)

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        if self.make_it_fit:
            n_params = len(
                [
                    name
                    for name, _ in itertools.chain(
                        self.named_parameters(), self.named_buffers()
                    )
                ]
            )
            for name, param in tqdm(
                itertools.chain(self.named_parameters(), self.named_buffers()),
                desc="Fitting old weights to new weights",
                total=n_params,
            ):
                if name not in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if new_shape != old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[
                                    i % old_shape[0], j % old_shape[1]
                                ]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        # print(
        #     f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        # )
        # if len(missing) > 0:
        #     print(f"Missing Keys:\n {missing}")
        # if len(unexpected) > 0:
        #     print(f"\nUnexpected Keys:\n {unexpected}")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        self.init_from_state_dict(sd, ignore_keys=ignore_keys, only_model=only_model)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            msg = f"Parameterization {self.parameterization} not yet supported"
            raise NotImplementedError(msg)

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(
        self, batch, N=8, n_row=2, *, sample=True, return_keys=None, **kwargs
    ):
        log = {}
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = []
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = [*params, self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


def _TileModeConv2DConvForward(
    self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor  # noqa
):
    if self.padding_modeX == self.padding_modeY:
        self.padding_mode = self.padding_modeX
        return self._orig_conv_forward(input, weight, bias)

    w1 = F.pad(input, self.paddingX, mode=self.padding_modeX)
    del input

    w2 = F.pad(w1, self.paddingY, mode=self.padding_modeY)
    del w1

    return F.conv2d(w2, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


class LatentDiffusion(DDPM):
    """main class."""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        unet_trainable=True,
        **kwargs,
    ):
        self.num_timesteps_cond = (
            1 if num_timesteps_cond is None else num_timesteps_cond
        )
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.unet_trainable = unet_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:  # noqa
            logger.exception("Bad num downs?")
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.cond_ids = None
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # store initial padding mode so we can switch to 'circular'
        # when we want tiled images
        # replace conv_forward with function that can do tiling in one direction
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m._initial_padding_mode = m.padding_mode
                m._orig_conv_forward = m._conv_forward
                m._conv_forward = _TileModeConv2DConvForward.__get__(m, nn.Conv2d)
        self.tile_mode(tile_mode=False)

    def tile_mode(self, tile_mode):
        """For creating seamless tiles."""
        tile_mode = tile_mode or ""
        tile_x = "x" in tile_mode
        tile_y = "y" in tile_mode
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_modeX = "circular" if tile_x else "constant"
                m.padding_modeY = "circular" if tile_y else "constant"
                if m.padding_modeY == m.padding_modeX:
                    m.padding_mode = m.padding_modeX
                m.paddingX = (
                    m._reversed_padding_repeated_twice[0],
                    m._reversed_padding_repeated_twice[1],
                    0,
                    0,
                )
                m.paddingY = (
                    0,
                    0,
                    m._reversed_padding_repeated_twice[2],
                    m._reversed_padding_repeated_twice[3],
                )

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                logger.debug("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                logger.debug(
                    f"Training {self.__class__.__name__} as an unconditional model."
                )
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc=""):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.mode()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            msg = f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            raise NotImplementedError(msg)
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(
            torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1
        )[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting,
                self.split_input_params["clip_min_tie_weight"],
                self.split_input_params["clip_max_tie_weight"],
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(
        self, x, kernel_size, stride, uf=1, df=1
    ):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = {
                "kernel_size": kernel_size,
                "dilation": 1,
                "padding": 0,
                "stride": stride,
            }
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = {
                "kernel_size": kernel_size,
                "dilation": 1,
                "padding": 0,
                "stride": stride,
            }
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = {
                "kernel_size": (kernel_size[0] * uf, kernel_size[0] * uf),
                "dilation": 1,
                "padding": 0,
                "stride": (stride[0] * uf, stride[1] * uf),
            }
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            Ly = (h - (kernel_size[0] * df)) // (stride[0] * df) + 1
            Lx = (w - (kernel_size[1] * df)) // (stride[1] * df) + 1

            unfold_params = {
                "kernel_size": (kernel_size[0] * df, kernel_size[1] * df),
                "dilation": 1,
                "padding": 0,
                "stride": (stride[0] * df, stride[1] * df),
            }

            unfold = torch.nn.Unfold(**unfold_params)

            fold_params = {
                "kernel_size": kernel_size,
                "dilation": 1,
                "padding": 0,
                "stride": stride,
            }
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params
            )

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox", "txt"]:
                    xc = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, (dict, list)):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, "pos_x": pos_x, "pos_y": pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        if False and hasattr(self, "split_input_params"):  # noqa
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(
                x_noisy, ks, stride
            )

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view(
                (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
            )  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if (
                self.cond_stage_key in ["image", "LR_image", "segmentation", "bbox_img"]
                and self.model.conditioning_key
            ):  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert len(c) == 1  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view(
                    (c.shape[0], -1, ks[0], ks[1], c.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == "coordinates_bbox":
                assert (
                    "original_image_size" in self.split_input_params
                ), "BoudingBoxRescaling is missing original_image_size"

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params["original_image_size"]
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left positions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [
                    (
                        rescale_latent
                        * stride[0]
                        * (patch_nr % n_patches_per_row)
                        / full_img_w,
                        rescale_latent
                        * stride[1]
                        * (patch_nr // n_patches_per_row)
                        / full_img_h,
                    )
                    for patch_nr in range(z.shape[-1])
                ]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [
                    (
                        x_tl,
                        y_tl,
                        rescale_latent * ks[0] / full_img_w,
                        rescale_latent * ks[1] / full_img_h,
                    )
                    for x_tl, y_tl in tl_patch_coordinates
                ]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [
                    torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(
                        self.device
                    )
                    for bbox in patch_limits
                ]  # list of length l with tensors of shape (1, 2)

                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), "cond must be dict to be fed into model"
                cut_cond = cond["c_crossattn"][0][..., :-2].to(self.device)

                adapted_cond = torch.stack(
                    [torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd]
                )
                adapted_cond = rearrange(adapted_cond, "l b n -> (l b) n")

                adapted_cond = self.get_learned_conditioning(adapted_cond)

                adapted_cond = rearrange(
                    adapted_cond, "(l b) n d -> l b n d", l=z.shape[-1]
                )

                cond_list = [{"c_crossattn": [e]} for e in adapted_cond]

            else:
                cond_list = [
                    cond for i in range(z.shape[-1])
                ]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [
                self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])
            ]
            assert not isinstance(
                output_list[0], tuple
            )  # todo can't deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]

        return x_recon

    def p_losses(self, x_start, cond, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        # t sometimes on wrong device. not sure why
        logvar_t = self.logvar[t.to(self.logvar.device)].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, _ = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device="cpu").to(x_start.device)

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        sampler = DPMPP2MSampler(self)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        uncond = kwargs.get("unconditional_conditioning")
        if uncond is None:
            uncond = self.get_unconditional_conditioning(batch_size, "")

        positive_conditioning = {
            "c_concat": [],
            "c_crossattn": [cond],
        }
        neutral_conditioning = {
            "c_concat": [],
            "c_crossattn": [uncond],
        }
        samples = sampler.sample(
            num_steps=ddim_steps,
            positive_conditioning=positive_conditioning,
            neutral_conditioning=neutral_conditioning,
            guidance_scale=kwargs.get("unconditional_guidance_scale", 5.0),
            shape=shape,
            batch_size=1,
        )

        return samples, []

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, (dict, list)):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            # todo: get null label from cond_stage_model
            raise NotImplementedError()
        c = repeat(c, "1 ... -> b ...", b=batch_size).to(self.device)
        return c

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample=True,
        ddim_steps=50,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        unconditional_guidance_scale=1.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = {}
        z, c, x, xrec, xc = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img(
                    (x.shape[2], x.shape[3]),
                    batch[self.cond_stage_key],
                    size=x.shape[2] // 25,
                )
                log["conditioning"] = xc
            elif self.cond_stage_key == "class_label":
                # xc = log_txt_as_img(
                #     (x.shape[2], x.shape[3]),
                #     batch["human_label"],
                #     size=x.shape[2] // 25,
                # )
                log["conditioning"] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = []
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if (
                quantize_denoised
                and not isinstance(self.first_stage_model, AutoencoderKL)
                and not isinstance(self.first_stage_model, IdentityFirstStage)
            ):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(
                        cond=c,
                        batch_size=N,
                        ddim=use_ddim,
                        ddim_steps=ddim_steps,
                        eta=ddim_eta,
                        quantize_denoised=True,
                    )
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # uc = torch.zeros_like(c)
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[
                    f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"
                ] = x_samples_cfg

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]  # noqa
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.0
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    eta=ddim_eta,
                    ddim_steps=ddim_steps,
                    x0=z[:N],
                    mask=mask,
                )
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_inpainting"] = x_samples
            log["mask"] = mask

            # outpaint
            mask = 1.0 - mask
            with ema_scope("Plotting Outpaint"):
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    eta=ddim_eta,
                    ddim_steps=ddim_steps,
                    x0=z[:N],
                    mask=mask,
                )
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(
                    c,
                    shape=(self.channels, self.image_size, self.image_size),
                    batch_size=N,
                )
            prog_row = self._get_denoise_row_from_list(
                progressives, desc="Progressive Generation"
            )
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log

            return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.unet_trainable == "attn":
            logger.info("Training only unet attention layers")
            for n, m in self.model.named_modules():
                if isinstance(m, CrossAttention) and n.endswith("attn2"):
                    params.extend(m.parameters())
        elif self.unet_trainable is True or self.unet_trainable == "all":
            logger.info("Training the full unet")
            params = list(self.model.parameters())
        else:
            msg = f"Unrecognised setting for unet_trainable: {self.unet_trainable}"
            raise ValueError(msg)

        if self.cond_stage_trainable:
            logger.info(
                f"{self.__class__.__name__}: Also optimizing conditioner params!"
            )
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            logger.info("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            logger.info("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm"]

    def forward(
        self, x, t, c_concat: Optional[list] = None, c_crossattn: Optional[list] = None
    ):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x, *c_concat], dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x, *c_concat], dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class LatentFinetuneDiffusion(LatentDiffusion):
    """
    Basis for different finetunas, such as inpainting or depth2image
    To disable finetuning mode, set finetune_keys to None.
    """

    def __init__(
        self,
        concat_keys: tuple,
        finetune_keys=(
            "model.diffusion_model.input_blocks.0.0.weight",
            "model_ema.diffusion_modelinput_blocks00weight",
        ),
        keep_finetune_dims=4,
        # if model was trained without concat mode before and we would like to keep these channels
        c_concat_log_start=None,  # to log reconstruction of c_concat codes
        c_concat_log_end=None,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(**kwargs)
        self.finetune_keys = finetune_keys
        self.concat_keys = concat_keys
        self.keep_dims = keep_finetune_dims
        self.c_concat_log_start = c_concat_log_start
        self.c_concat_log_end = c_concat_log_end
        if self.finetune_keys is not None:
            assert ckpt_path is not None, "can only finetune from a given checkpoint"
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        return self.init_from_state_dict(
            sd, ignore_keys=ignore_keys, only_model=only_model
        )

    def init_from_state_dict(self, sd, ignore_keys=(), only_model=False):
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {k} from state_dict.")
                    del sd[k]

            # make it explicit, finetune by including extra input channels
            if self.finetune_keys is not None and k in self.finetune_keys:
                new_entry = None
                for name, param in self.named_parameters():
                    if name in self.finetune_keys:
                        print(
                            f"modifying key '{name}' and keeping its original {self.keep_dims} (channels) dimensions only"
                        )
                        new_entry = torch.zeros_like(param)  # zero init
                assert (
                    new_entry is not None
                ), "did not find matching parameter to modify"
                new_entry[:, : self.keep_dims, ...] = sd[k]
                sd[k] = new_entry

        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        # print(
        #     f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        # )
        # if len(missing) > 0:
        #     print(f"Missing Keys: {missing}")
        # if len(unexpected) > 0:
        #     print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample=True,
        ddim_steps=200,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=True,
        unconditional_guidance_scale=1.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = {}
        z, c, x, xrec, xc = self.get_input(
            batch, self.first_stage_key, bs=N, return_first_stage_outputs=True
        )
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                # xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["class_label", "cls"]:
                # xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if not (self.c_concat_log_start is None and self.c_concat_log_end is None):
            log["c_concat_decoded"] = self.decode_first_stage(
                c_cat[:, self.c_concat_log_start : self.c_concat_log_end]
            )

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = []
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(
                    cond={"c_concat": [c_cat], "c_crossattn": [c]},
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(
                N, unconditional_guidance_label
            )
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(
                    cond={"c_concat": [c_cat], "c_crossattn": [c]},
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc_full,
                )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[
                    f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"
                ] = x_samples_cfg

        return log


class LatentInpaintDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys

    @torch.no_grad()
    def get_input(
        self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False
    ):
        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for inpainting"
        z, c, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=bs,
        )

        assert self.concat_keys is not None
        c_cat = []
        for ck in self.concat_keys:
            cc = (
                rearrange(batch[ck], "b h w c -> b c h w")
                .to(memory_format=torch.contiguous_format)
                .float()
            )
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            bchw = z.shape
            if ck != self.masked_image_key:
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds


class LatentDepth2ImageDiffusion(LatentFinetuneDiffusion):
    """
    condition on monocular depth estimation.
    """

    def __init__(self, depth_stage_config, concat_keys=("midas_in",), **kwargs):
        super().__init__(concat_keys=concat_keys, **kwargs)
        self.depth_model = instantiate_from_config(depth_stage_config)
        self.depth_stage_key = concat_keys[0]

    @torch.no_grad()
    def get_input(
        self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False
    ):
        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for depth2img"
        z, c, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=bs,
        )

        assert self.concat_keys is not None
        assert len(self.concat_keys) == 1
        c_cat = []
        for ck in self.concat_keys:
            cc = batch[ck]
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            cc = self.depth_model(cc)
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )

            depth_min, depth_max = torch.amin(
                cc, dim=[1, 2, 3], keepdim=True
            ), torch.amax(cc, dim=[1, 2, 3], keepdim=True)
            cc = 2.0 * (cc - depth_min) / (depth_max - depth_min + 0.001) - 1.0
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        depth = self.depth_model(args[0][self.depth_stage_key])
        depth_min, depth_max = torch.amin(
            depth, dim=[1, 2, 3], keepdim=True
        ), torch.amax(depth, dim=[1, 2, 3], keepdim=True)
        log["depth"] = 2.0 * (depth - depth_min) / (depth_max - depth_min) - 1.0
        return log


class LatentUpscaleFinetuneDiffusion(LatentFinetuneDiffusion):
    """
    condition on low-res image (and optionally on some spatial noise augmentation).
    """

    def __init__(
        self,
        concat_keys=("lr",),
        reshuffle_patch_size=None,
        low_scale_config=None,
        low_scale_key=None,
        **kwargs,
    ):
        super().__init__(concat_keys=concat_keys, **kwargs)
        self.reshuffle_patch_size = reshuffle_patch_size
        self.low_scale_model = None
        if low_scale_config is not None:
            print("Initializing a low-scale model")
            assert low_scale_key is not None
            self.instantiate_low_stage(low_scale_config)
            self.low_scale_key = low_scale_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        self.low_scale_model = model.eval()
        self.low_scale_model.train = disabled_train
        for param in self.low_scale_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_input(
        self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False
    ):
        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for upscaling-ft"
        z, c, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=bs,
        )

        assert self.concat_keys is not None
        assert len(self.concat_keys) == 1
        # optionally make spatial noise_level here
        c_cat = []
        noise_level = None
        for ck in self.concat_keys:
            cc = batch[ck]
            cc = rearrange(cc, "b h w c -> b c h w")
            if self.reshuffle_patch_size is not None:
                assert isinstance(self.reshuffle_patch_size, int)
                cc = rearrange(
                    cc,
                    "b c (p1 h) (p2 w) -> b (p1 p2 c) h w",
                    p1=self.reshuffle_patch_size,
                    p2=self.reshuffle_patch_size,
                )
            if bs is not None:
                cc = cc[:bs]
                cc = cc.to(self.device)
            if self.low_scale_model is not None and ck == self.low_scale_key:
                cc, noise_level = self.low_scale_model(cc)
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        if noise_level is not None:
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c], "c_adm": noise_level}
        else:
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        log["lr"] = rearrange(args[0]["lr"], "b h w c -> b c h w")
        return log
