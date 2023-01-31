from .unet_model import UNet
import torch
import torch.nn as nn
import importlib
from torchmetrics.functional import accuracy
import torch.nn.functional as F

dice_score = importlib.import_module("brp-algal-bloom-forecasting.ralvarezlucend.utils.dice_score")
focal_loss = importlib.import_module("brp-algal-bloom-forecasting.ralvarezlucend.utils.focal_loss")


class UNetClassification(UNet):

    def _compute_metrics(self, batch):
        input_images, labeled_images, _, _ = self._process_batch(batch)
        output_images = self(input_images)

        # # compute effective number of samples.
        # samples_per_class = torch.bincount(labeled_images.flatten())
        # if samples_per_class.size() == self.hparams.n_classes:
        #     beta = self.hparams.beta
        #     effective_num = 1.0 - torch.pow(beta, samples_per_class)
        #     weight = (1.0 - beta) / effective_num
        # else:
        #     weight = torch.ones(self.hparams.n_classes).to(input_images)
        
        # weight = weight / weight.sum()  # normalize weights.

        # weight = torch.ones(self.hparams.n_classes).to(input_images)
        # criterion = focal_loss.FocalLoss(weight=weight, gamma=self.hparams.gamma)
        # criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=0).double()
        # loss = criterion(input=output_images, target=labeled_images)
        loss = dice_score.dice_loss(
             F.softmax(output_images, dim=1).float(),
             F.one_hot(labeled_images, self.hparams.n_classes).permute(0, 3, 1, 2).float(),
             multiclass=True
        )

        # compute accuracy.
        acc = accuracy(preds=output_images, target=labeled_images, ignore_index=0)

        return loss, acc

    def training_step(self, batch, batch_idx):
        self.train()
        loss, acc = self._compute_metrics(batch)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_train_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log('avg/train_loss', avg_train_loss)
        self.log('avg/train_acc', avg_train_acc)

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss, acc = self._compute_metrics(batch)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('avg/val_loss', avg_val_loss)
        self.log('avg/val_acc', avg_val_acc)

    def predict_step(self, batch, batch_idx):
        self.eval()
        input_image, labeled_image, mask, classification_info = self._process_batch(batch)
        output_image = self(input_image)
        probs = F.softmax(output_image, dim=1)[0]
        return probs, labeled_image.squeeze(), mask.squeeze().numpy(), classification_info
