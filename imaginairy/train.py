import datetime
import logging
import os
import signal
import time
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor

try:
    from pytorch_lightning.strategies import DDPStrategy
except ImportError:
    # let's not break all of imaginairy just because a training import doesn't exist in an older version of PL
    # Use >= 1.6.0 to make this work
    DDPStrategy = None
import contextlib

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from imaginairy import config
from imaginairy.model_manager import get_diffusion_model
from imaginairy.training_tools.single_concept import SingleConceptDataset
from imaginairy.utils import get_device, instantiate_from_config

mod_logger = logging.getLogger(__name__)

referenced_by_string = [LearningRateMonitor]


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset."""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, SingleConceptDataset):
        # split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        # dataset.sample_ids = dataset.valid_ids[
        #     worker_id * split_size : (worker_id + 1) * split_size
        # ]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
        num_val_workers=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader
            )
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.datasets = None

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = {
            k: instantiate_from_config(c) for k, c in self.dataset_configs.items()
        }
        if self.wrap:
            self.datasets = {k: WrappedDataset(v) for k, v in self.datasets.items()}

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets["train"], SingleConceptDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            pass
        else:
            pass
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )

    def _val_dataloader(self, shuffle=False):
        if (
            isinstance(self.datasets["validation"], SingleConceptDataset)
            or self.use_worker_init_fn
        ):
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_val_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets["train"], SingleConceptDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        is_iterable_dataset = False

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _predict_dataloader(self, shuffle=False):
        if (
            isinstance(self.datasets["predict"], SingleConceptDataset)
            or self.use_worker_init_fn
        ):
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
        )


class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            mod_logger.info("Stopping execution and saving final checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                with contextlib.suppress(FileNotFoundError):
                    os.rename(self.logdir, dst)


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_all_val=False,
        concept_label=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {}
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val
        self.concept_label = concept_label

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "logs", "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = (
                f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # always generate the concept label
        batch["txt"][0] = self.concept_label

        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(check_idx)
        if (
            should_log
            and (batch_idx % self.batch_freq == 0)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None
            )
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0 and (
            check_idx > 0 or self.log_first_step
        ):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if (
            hasattr(pl_module, "calibrate_grad_norm")
            and (pl_module.calibrate_grad_norm and batch_idx % 25 == 0)
            and batch_idx > 0
        ):
            self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        if "cuda" in get_device():
            torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
            torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if "cuda" in get_device():
            torch.cuda.synchronize(trainer.strategy.root_device.index)
            max_memory = (
                torch.cuda.max_memory_allocated(trainer.strategy.root_device.index)
                / 2**20
            )
            epoch_time = time.time() - self.start_time

            try:
                max_memory = trainer.training_type_plugin.reduce(max_memory)
                epoch_time = trainer.training_type_plugin.reduce(epoch_time)

                rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
                rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
            except AttributeError:
                pass


def train_diffusion_model(
    concept_label,
    concept_images_dir,
    class_label,
    class_images_dir,
    weights_location=config.DEFAULT_MODEL,
    logdir="logs",
    learning_rate=1e-6,
    accumulate_grad_batches=32,
    resume=None,
):
    """
    Train a diffusion model on a single concept.

    accumulate_grad_batches used to simulate a bigger batch size - https://arxiv.org/pdf/1711.00489.pdf
    """
    if DDPStrategy is None:
        msg = "Please install pytorch-lightning>=1.6.0 to train a model"
        raise ImportError(msg)

    batch_size = 1
    seed = 23
    num_workers = 1
    num_val_workers = 0
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # noqa: DTZ005
    logdir = os.path.join(logdir, now)

    ckpt_output_dir = os.path.join(logdir, "checkpoints")
    cfg_output_dir = os.path.join(logdir, "configs")
    seed_everything(seed)
    model = get_diffusion_model(
        weights_location=weights_location, half_mode=False, for_training=True
    )._model
    model.learning_rate = learning_rate * accumulate_grad_batches * batch_size

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "imaginairy.train.SetupCallback",
            "params": {
                "resume": False,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckpt_output_dir,
                "cfgdir": cfg_output_dir,
            },
        },
        "image_logger": {
            "target": "imaginairy.train.ImageLogger",
            "params": {
                "batch_frequency": 10,
                "max_images": 1,
                "clamp": True,
                "increase_log_steps": False,
                "log_first_step": True,
                "log_all_val": True,
                "concept_label": concept_label,
                "log_images_kwargs": {
                    "use_ema_scope": True,
                    "inpaint": False,
                    "plot_progressive_rows": False,
                    "plot_diffusion_rows": False,
                    "N": 1,
                    "unconditional_guidance_scale:": 7.5,
                    "unconditional_guidance_label": [""],
                    "ddim_steps": 20,
                },
            },
        },
        "learning_rate_logger": {
            "target": "imaginairy.train.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            },
        },
        "cuda_callback": {"target": "imaginairy.train.CUDACallback"},
    }

    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckpt_output_dir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 50,
            "save_top_k": -1,
            "monitor": None,
        },
    }

    modelckpt_cfg = OmegaConf.create(default_modelckpt_cfg)
    default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    callbacks_cfg = OmegaConf.create(default_callbacks_cfg)

    dataset_config = {
        "concept_label": concept_label,
        "concept_images_dir": concept_images_dir,
        "class_label": class_label,
        "class_images_dir": class_images_dir,
        "image_transforms": [
            {
                "target": "torchvision.transforms.Resize",
                "params": {"size": 512, "interpolation": 3},
            },
            {"target": "torchvision.transforms.RandomCrop", "params": {"size": 512}},
        ],
    }

    data_module_config = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_val_workers": num_val_workers,
        "train": {
            "target": "imaginairy.training_tools.single_concept.SingleConceptDataset",
            "params": dataset_config,
        },
    }
    trainer = Trainer(
        benchmark=True,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        strategy=DDPStrategy(),
        callbacks=[instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg],
        gpus=1,
        default_root_dir=".",
    )
    trainer.logdir = logdir

    data = DataModuleFromConfig(**data_module_config)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()

    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            mod_logger.info("Summoning checkpoint.")
            ckpt_path = os.path.join(ckpt_output_dir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    signal.signal(signal.SIGUSR1, melk)
    try:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    finally:
        mod_logger.info(trainer.profiler.summary())
