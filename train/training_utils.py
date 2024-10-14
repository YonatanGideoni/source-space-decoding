from copy import deepcopy

import torch
import torchmetrics.classification
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from consts import device
from data.data_aug_utils import DataAug, mixup
from train.models import get_model


def get_trainer(args, log_model: bool = False):
    wandb_logger = WandbLogger(log_model=log_model)
    early_stop_callback = EarlyStopping(monitor="val_acc", patience=10, verbose=False, mode="max", min_delta=0.001)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")
    trainer = Trainer(max_epochs=args.num_epochs, logger=wandb_logger,
                      callbacks=[early_stop_callback, checkpoint_callback])
    return trainer


class SpeechDetectionModel(LightningModule):
    def __init__(self, input_dim: int, args, n_subjects=None):
        super(SpeechDetectionModel, self).__init__()
        self.args = args
        self.save_hyperparameters()
        Model = get_model(args)
        self.model = Model(input_dim=input_dim, args=args, n_subjects=n_subjects)
        self.criterion = nn.BCEWithLogitsLoss()
        self.transform = DataAug(args)
        # as accuracy is a global and not batchwise metric it needs to be accumulated. if we do this naively it'll
        # be imprecise and give totally wrong results given small enough batches
        acc_fn = torchmetrics.classification.Accuracy(average='macro', num_classes=2, task='multiclass')
        self.train_metrics = torchmetrics.MetricCollection({
            'train_acc': deepcopy(acc_fn)
        })
        self.val_metrics = torchmetrics.MetricCollection({
            'val_acc': deepcopy(acc_fn)
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_acc': deepcopy(acc_fn)
        })

        self.voxel_mask = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        y = batch['y']
        outputs = self(batch).squeeze(dim=1)
        if y.dim() == 2:
            y = y[:, 1]  # one hot encoding/mixup
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_accuracy(outputs, y, stage='train')

        return loss

    def log_accuracy(self, outputs, labels, stage='val'):
        preds = (outputs >= 0.).int()
        if stage == 'train':
            if self.args.mixup_alpha >= 0:
                return
            self.train_metrics['train_acc'](preds, labels)
        elif stage == 'val':
            self.val_metrics['val_acc'](preds, labels)
        elif stage == 'test':
            self.test_metrics['test_acc'](preds, labels)

    def validation_step(self, batch, batch_idx, loss_name='val'):
        y = batch['y']
        outputs = self(batch).squeeze(dim=1)
        loss = self.criterion(outputs, y.float())
        self.log(f'{loss_name}_loss', loss, on_epoch=True, prog_bar=False, logger=True)
        self.log_accuracy(outputs, y, stage=loss_name)

    def on_train_epoch_end(self) -> None:
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics, on_epoch=True, prog_bar=True, logger=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, on_epoch=True, prog_bar=True, logger=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, loss_name='test')

    def on_test_epoch_end(self):
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics, on_epoch=True, prog_bar=False, logger=True)
        self.test_metrics.reset()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch['x'], batch['y']
        if self.trainer.training:
            x, y = self.transform(x, y)  # GPU/Batched data augmentation

            if self.args.mixup_alpha >= 0:
                x, y = mixup(x, y, alpha=self.args.mixup_alpha, n_classes=2)

        if self.voxel_mask is not None:
            n_voxels = self.voxel_mask.shape[0]
            assert n_voxels == x.shape[1] // 3  # 3 vector components
            exp_mask = (torch.tensor(self.voxel_mask, dtype=torch.bool, device=device)
                        .unsqueeze(-1).repeat(1, 3).view(n_voxels * 3).unsqueeze(0).repeat(x.shape[0], 1)).unsqueeze(-1)
            x = x.masked_fill(exp_mask, 0)

        batch['x'], batch['y'] = x, y
        return batch

    def forward(self, x):
        return self.model(x)
