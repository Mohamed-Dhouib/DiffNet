"""Lightning training module: losses, optimizer and LR schedules of DiffNet."""
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from act_loss import activation_contrast_loss, act_loss_weight
from model import DocMDetector
from threshold import macro_soft_f1_loss, soft_f1_schedule

WEIGHT_DECAY = 0.01
MIN_LR = 5e-6            # cosine floor of the network LR (after the warmup); config min_lr overrides it
OFFSET_LR = 1e-3         # threshold-offset head: own cosine schedule OFFSET_LR -> OFFSET_MIN_LR, no warmup / decay
OFFSET_MIN_LR = 1e-4
SOFT_F1_WEIGHT = 0.5     # positive-image macro soft-F1 on the thresholded map
SOFT_F1_RAMP_FRACTION = 0.05
SOFT_F1_BANDWIDTH_FINAL = 0.1
SOFT_F1_ANNEAL_FRACTION = 0.8
ACT_LAMBDA_MAX = 0.005   # activation-contrast loss weight: 0 for the first 10% of the steps, then linear ramp
ACT_DELAY_FRACTION = 0.1
ACT_RAMP_FRACTION = 0.1
ACT_TOP_FRACTION = 0.1
ACT_D_TARGET = [0.6, 0.6, 0.6, 0.6]       # contrast a/b = 4 at every stage: (4 - 1) / (4 + 1)
ACT_STAGE_WEIGHTS = [0.25, 0.5, 1, 1]     # strides 4, 8, 16, 32
LOG_EVERY = 500


def _worker_init_threads(_):
    import os, cv2, torch
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['BLOSC_NTHREADS'] = '1'


    torch.set_num_threads(1)


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation."""
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
      
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )


class DocMDetectorPLModule(pl.LightningModule):
    """L = focal(seg, raw logits) + class_coef * focal(image) + w(t) soft-F1(thresholded map) + lambda(t) L_act.

    Config switches (Doc Protocol: all defaults; Syn2Real-TDoc: keeps only the activation-contrast loss):
    learned_threshold (default true): train the threshold offset with the soft-F1 term; false fixes the offset at 0
    and drops the soft-F1 term. regularisation (default true): stochastic depth and decoder dropout. min_lr (default
    MIN_LR): cosine floor of the network LR."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learned_threshold = config.get("learned_threshold", True)
        # The activation maps are requested only once lambda > 0 (on_train_batch_start, one switch).
        self.act_maps_on = False

        self.register_buffer(
            "seg_loss_weight",
            torch.tensor([1.0, config.loss_pos_weight], dtype=torch.float32)
        )
        self.register_buffer(
            "class_loss_weight",
            torch.tensor([1.0, 1.0], dtype=torch.float32)
        )

        self.Focal_fn=FocalLoss(gamma=2.,weight=self.seg_loss_weight)
        self.Focal_fn_class=FocalLoss(gamma=2.,weight=self.class_loss_weight)

        # On resume, Lightning restores all weights from the checkpoint: no initialisation is loaded.
        resume = self.config.get("resume_from_checkpoint_path", None)
        self.model = DocMDetector(
            input_size=self.config.input_size,
            pretrained_model_name_or_path=None if resume else self.config.pretrained_model_name_or_path,
            data_aug_config=self.config.data_aug_config,
            load_imagenet_weights=not resume,
            regularisation=self.config.get("regularisation", True),
        )
        if not self.learned_threshold:
            nn.init.zeros_(self.model.encoder.model.offset_head.beta)  # offset 0: the cut is the plain 0.5
      
    def training_step(self, batch, batch_idx):
        if batch_idx==0 or not hasattr(self, 'loss'):
            self.loss=[]
            self.loss_class=[]

        image_tensors, masks, dcts, qt = batch
        
        if self.act_maps_on:
            pred, output_class, raw_pred, act_maps = self.model.encoder(image_tensors, dcts, qt, return_raw=True)
        else:
            pred, output_class, raw_pred = self.model.encoder(image_tensors, dcts, qt, return_raw=True)
 
        classification_gt = torch.any(input=(masks > 0).flatten(start_dim=1),dim=1).long()


        masks=masks.squeeze(1).long() 

        # The focal loss supervises the raw logits; the learned offset only moves the operating point.
        loss=self.Focal_fn(raw_pred,masks)

        loss_class=self.Focal_fn_class(output_class,classification_gt)

        final_loss=loss+loss_class*self.config.class_coef

        if self.act_maps_on:
            final_loss = final_loss + self._act_loss_term(act_maps, masks, batch_idx)

        if self.learned_threshold:
            head = self.model.encoder.model.offset_head
            coefficient, bandwidth = soft_f1_schedule(  # one-based step
                self.global_step + 1, self.trainer.estimated_stepping_batches,
                weight=SOFT_F1_WEIGHT,
                ramp_fraction=SOFT_F1_RAMP_FRACTION,
                bandwidth_final=SOFT_F1_BANDWIDTH_FINAL,
                anneal_fraction=SOFT_F1_ANNEAL_FRACTION)
            loss_f1, n_positive = macro_soft_f1_loss(pred, masks, head.anchor, bandwidth)
            final_loss = final_loss + coefficient * loss_f1
            if batch_idx % LOG_EVERY == 0:  # rank-local (loss_f1 itself is the global value)
                self.log_dict({'l_softf1': loss_f1.detach(), 'softf1_coef': coefficient}, on_step=True,
                              on_epoch=False)

        self.loss.append(loss.item())
        self.loss_class.append(loss_class.item())
      
        if len(self.loss)>5000:
            self.loss=self.loss[1:]  
            self.loss_class=self.loss_class[1:] 
       
        if (batch_idx) % LOG_EVERY == 0:
            self.log_dict({"l_seg": sum(self.loss)/len(self.loss),"l_c" :sum(self.loss_class)/len(self.loss_class)}, sync_dist=True , prog_bar=True)
            if self.global_rank == 0:
                cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return final_loss

    def _act_loss_term(self, act_maps, masks, batch_idx):
        """lambda(t) L_act; logs lambda, L_act and the per-stage mean contrast d_s over the valid images."""
        lam = self._act_lambda()
        loss_aux, d, _, valid = activation_contrast_loss(act_maps, masks, ACT_D_TARGET, ACT_TOP_FRACTION,
                                                         ACT_STAGE_WEIGHTS)
        if batch_idx % LOG_EVERY == 0:
            stats = {'act_lambda': lam, 'l_act': loss_aux.detach()}
            d, valid = d.detach(), valid.float()
            for s in range(d.shape[1]):
                stats[f'act_d{s + 1}'] = (d[:, s] * valid[:, s]).sum() / valid[:, s].sum().clamp(min=1)
            self.log_dict(stats, on_step=True, on_epoch=False)
        return lam * loss_aux

    def _act_lambda(self):
        return act_loss_weight(self.global_step, self.trainer.estimated_stepping_batches, ACT_LAMBDA_MAX,
                               ACT_DELAY_FRACTION, ACT_RAMP_FRACTION)

    def on_train_batch_start(self, batch, batch_idx):
        if not self.act_maps_on and self._act_lambda() > 0:
            # One switch, never back (lambda is non-decreasing): the compiled encoder recompiles once here. A resume
            # decides again from the restored global_step at its first batch.
            self.model.encoder.model.set_return_activation_maps(True)
            self.act_maps_on = True

    def configure_optimizers(self):
        assert self.config.get("max_epochs", -1)>0

        samples_per_cycle = len(self.trainer.datamodule.train_dataset)
        max_iter = (self.config.max_epochs * samples_per_cycle) / (
            self.config.train_batch_size * torch.cuda.device_count() * self.trainer.num_nodes
        )
        max_iter=int(max_iter)
        print(f"max iter is set to {max_iter}")
        
        head_params = list(self.model.encoder.model.offset_head.parameters())
        for _,param in self.model.named_parameters():
            param.requires_grad=True
        if not self.learned_threshold:
            for param in head_params:
                param.requires_grad = False  # offset stays at 0

        parameters = [p for _, p in self.model.named_parameters() if p.requires_grad]
        parameters = [{'params': [p for p in parameters if all(p is not q for q in head_params)]}]
        if self.learned_threshold:
            parameters.append({'params': head_params, 'lr': OFFSET_LR, 'weight_decay': 0.0})
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.lr,
            weight_decay=WEIGHT_DECAY
        )

        total_steps = self.trainer.estimated_stepping_batches
        head_min_factor = OFFSET_MIN_LR / OFFSET_LR
        def offset_schedule(step):
            progress = min(max(step, 0) / max(1, total_steps), 1.0)
            return head_min_factor + 0.5 * (1 - head_min_factor) * (1 + math.cos(math.pi * progress))
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps,
                                               min_lr=self.config.get("min_lr", MIN_LR),
                                               offset_schedule=offset_schedule if self.learned_threshold else None),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps, min_lr, offset_schedule):
        """Linear warmup then cosine decay, floored at min_lr after the warmup (network group); the offset head
        group, when present, follows offset_schedule."""
        base_lr = float(optimizer.param_groups[0].get('lr', 1.0))
        min_factor = 0.0
        if base_lr > 0.0:
            min_factor = float(min_lr) / base_lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, [lr_lambda, offset_schedule] if offset_schedule else lr_lambda)


class DocMDetectorDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_size = self.config.train_batch_size
        self.train_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=_worker_init_threads,
            )
