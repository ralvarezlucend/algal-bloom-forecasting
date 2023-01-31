from .unet_model import UNet
import torch


class UNetRegression(UNet):
    def _compute_metrics(self, batch):
        input_images, labeled_images, mask, _ = self._process_batch(batch)
        output_images = self(input_images)

        # Compute MSE Loss.
        output_images = output_images[:, -1, :, :]  # take last element.
        out = (output_images[~mask] - labeled_images[~mask]) ** 2
        loss = out.mean()
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self._compute_metrics(batch)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg/train_loss', avg_train_loss)

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self._compute_metrics(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg/val_loss', avg_val_loss)

    def predict_step(self, batch, batch_idx):
        self.eval()
        input_image, labeled_image, mask, classification_info = self._process_batch(batch)
        output_image = self(input_image)
        output_image = self._reverse_preprocess(output_image)
        labeled_image = self._reverse_preprocess(labeled_image)
        return output_image.squeeze(), labeled_image.squeeze(), mask.squeeze().numpy(), classification_info