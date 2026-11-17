import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLOSC_NTHREADS'] = '1'
os.environ["PL_FAULT_TOLERANT_TRAINING"] = "0"   # no autosave blobs on crash/timeout
os.environ["LIGHTNING_FAULT_TOLERANT"] = "0"     # Fabric variant; harmless if unused

import argparse
from os.path import basename
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
from util import DocMDetectorDataset
from lightning_module import DocMDetectorDataPLModule, DocMDetectorPLModule
from pytorch_lightning.callbacks import TQDMProgressBar
import shutil
import datetime
from pytorch_lightning.strategies import DDPStrategy
import ast
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from datetime import timedelta

def setup_rank_caches() -> tuple[str, str]:
    """Per-node/per-rank TorchInductor and Triton cache directories, exported for this process."""

    home = Path.home()
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    base_inductor = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", xdg_cache / "torch" / "inductor"))
    base_triton   = Path(os.environ.get("TRITON_CACHE_DIR", xdg_cache / "triton"))

    node = os.environ.get("NODE_RANK") or "0"
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"

    sub = f"node{node}_rank{rank}"
    final_inductor = (base_inductor / sub).resolve()
    final_triton   = (base_triton / sub).resolve()

    final_inductor.mkdir(parents=True, exist_ok=True)
    final_triton.mkdir(parents=True, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(final_inductor)
    os.environ["TRITON_CACHE_DIR"] = str(final_triton)

    print(f"TORCHINDUCTOR_CACHE_DIR={final_inductor}")
    print(f"TRITON_CACHE_DIR={final_triton}")

    return str(final_inductor), str(final_triton)


cols, rows = shutil.get_terminal_size(fallback=(120, 30))
term_width = cols 

class CustomTqdm(TQDMProgressBar):
    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.ncols = term_width
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ncols = term_width
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ncols = term_width
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.ncols = term_width
        return bar

def epoch_checkpoints(max_epochs):
    """The epochs kept as epoch<N>.pth for the test evals: the last one and the two before it, every 5
    (30 epochs: 20, 25, 30)."""
    return [e for e in range(max_epochs - 10, max_epochs + 1, 5) if e >= 1]


class CheckpointAtTrainEnd(pl.Callback):
    def __init__(self, filename="periodic.pth"):
        self.filename = filename

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        ckpt_dir = trainer.checkpoint_callback.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, self.filename)
        print(f"saving on {ckpt_path}")
        checkpoint = trainer.model.state_dict()
        torch.save(checkpoint, ckpt_path)
        # Also epoch<N>.pth (same format) after a complete epoch N of epoch_checkpoints.
        epoch = trainer.current_epoch + 1
        loop = trainer.fit_loop.epoch_loop
        complete = loop.batch_progress.is_last_batch or loop.batch_idx + 1 == trainer.num_training_batches
        if epoch in epoch_checkpoints(trainer.max_epochs) and complete:
            print(f"saving on {os.path.join(ckpt_dir, f'epoch{epoch}.pth')}")
            torch.save(checkpoint, os.path.join(ckpt_dir, f"epoch{epoch}.pth"))

class LastCheckpoint(pl.Callback):
    """<checkpoints>/last.ckpt (weights, optimizer, LR scheduler and loop state) after every completed epoch,
    written atomically, so an interrupted run resumes from its last completed epoch."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        loop = trainer.fit_loop.epoch_loop
        if not (loop.batch_progress.is_last_batch or loop.batch_idx + 1 == trainer.num_training_batches):
            return
        ckpt_dir = trainer.checkpoint_callback.dirpath
        if trainer.is_global_zero:
            os.makedirs(ckpt_dir, exist_ok=True)
        tmp = os.path.join(ckpt_dir, "last.ckpt.tmp")
        trainer.save_checkpoint(tmp)  # on every rank (it synchronizes); rank 0 writes the file
        if trainer.is_global_zero:
            os.replace(tmp, os.path.join(ckpt_dir, "last.ckpt"))

class CheckpointEveryNSteps(pl.Callback):


    def __init__(
        self,
        steps_per_checkpoint=1000,
    ):

        self.steps_per_checkpoint = steps_per_checkpoint

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, _, a, b, batch_idx):
        if (batch_idx) % self.steps_per_checkpoint== 0 and batch_idx!=0 :
            print("saving periodically")
            filename = "periodic.pth"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            os.makedirs(trainer.checkpoint_callback.dirpath,exist_ok=True)
            print("saving on"+str(ckpt_path))
            checkpoint=trainer.model.state_dict()
            torch.save(checkpoint, ckpt_path)




@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):

    model_module = DocMDetectorPLModule(config)
    data_module = DocMDetectorDataPLModule(config)

    data_module.train_dataset = DocMDetectorDataset(
        dataset_name_or_path=config.dataset_name_or_path,
        seg_model=model_module.model,
        split="train",
        config=config,
    )

    logger = TensorBoardLogger(
        save_dir=config.result_path,
        name=config.exp_name,
        version=config.exp_version,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    ckpt_dir = Path(config.result_path) / config.exp_name / config.exp_version

    # periodic.pth (every steps_per_checkpoint batches and at every epoch end), epoch<N>.pth (last three multiples
    # of 5), last.ckpt (full state, for --resume_from_checkpoint_path).
    callbacks_list=[lr_callback,CustomTqdm(),CheckpointAtTrainEnd(),LastCheckpoint(),
                    CheckpointEveryNSteps(config.steps_per_checkpoint)]

    print(f"Training will run on {torch.cuda.device_count()} GPUs.")

    trainer = pl.Trainer(
        num_nodes=int(os.environ.get("NUM_NODES", "1")),
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False,timeout=timedelta(minutes=60)), # long timeout: compilation
        accelerator="auto",
        max_epochs=config.max_epochs,
        default_root_dir=ckpt_dir,
        val_check_interval=1.0,
        check_val_every_n_epoch=100,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks_list,
        accumulate_grad_batches=1,
    )
    

    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.recompile_limit = 512
    torch._dynamo.config.cache_size_limit = 512
    model_module.model.encoder =  torch.compile(model_module.model.encoder, mode="max-autotune-no-cudagraphs", dynamic=False )


    trainer.fit(model_module, data_module, ckpt_path=config.get("resume_from_checkpoint_path",None) )


if __name__ == "__main__":
    setup_rank_caches()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    # Resume from a last.ckpt (not a config key, so parsed here).
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()


    config = Config(args.config)
    config.argv_update(left_argv)
    torch.set_float32_matmul_precision("medium")

    config.exp_name = basename(args.config).split(".")[0]
    
    # SEED environment variable overrides the config seed
    seed_override = os.environ.get("SEED", None)
    if seed_override is not None:
        config.seed = int(seed_override)
        print(f"Seed overridden to {config.seed} (from SEED env var)")
    elif "seed" not in config:
        config.seed = 2025  # Default if not in config
    
    # Auto-append seed to exp_version if not already there
    base_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version
    if f"_seed{config.seed}" not in base_version:
        config.exp_version = f"{base_version}_seed{config.seed}"
    else:
        config.exp_version = base_version
    
    config.save_path_=os.path.join(config.result_path,config.exp_name,config.exp_version)

    
    dataset_name_or_path = ast.literal_eval(os.environ.get("DATASET_NAME_OR_PATH",str(config["dataset_name_or_path"])))
    print(f"Training on the following dataset input {dataset_name_or_path}")
    config["dataset_name_or_path"]=dataset_name_or_path


    pretrained_model_name_or_path=os.environ.get("PRETRAINED_PATH",config["pretrained_model_name_or_path"])
    config["pretrained_model_name_or_path"]=pretrained_model_name_or_path
    print(f"pretrained_model_name_or_path is set to {pretrained_model_name_or_path}")
  
    
    if args.resume_from_checkpoint_path:
        config.resume_from_checkpoint_path = args.resume_from_checkpoint_path

    save_config_file(config, Path(config.result_path) / config.exp_name / config.exp_version)
    print(config)
    pl.seed_everything(config.seed, workers=True)
    train(config)
