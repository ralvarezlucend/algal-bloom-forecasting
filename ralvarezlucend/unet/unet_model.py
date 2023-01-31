""" Full assembly of the parts to form the complete network """
import importlib
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler, Units
from .unet_parts import *
from pytorch_lightning import seed_everything, LightningModule
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

rionegrodata = importlib.import_module("brp-algal-bloom-forecasting.rionegrodata")
data_analysis = importlib.import_module("brp-algal-bloom-forecasting.ralvarezlucend.utils.analysis")


class UNet(LightningModule):
    def __init__(self,
                 root: str,
                 reservoir: str,
                 window_size: int,
                 prediction_horizon: int,
                 n_bands: int,
                 n_classes: int,
                 bins: list,
                 train_samples: int,
                 batch_size: int,
                 gamma: int,
                 beta: int,
                 num_workers: int,
                 overfit: bool,
                 bilinear: bool,
                 config: dict,
                 clip_value: int
                 ):

        super(UNet, self).__init__()
        self.learning_rate = config['lr']

        # set a seed to get deterministic results.
        if overfit:
            seed_everything(42, workers=True)

        # Load dataset
        self.dataset = rionegrodata.RioNegroData(
            root=root,
            reservoir=reservoir,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            input_size=224,
        )

        # make analysis of data.
        analysis = data_analysis.DataAnalysis(rionegro_dst=self.dataset, clip_value=clip_value)
        self._MEAN, self._STD, self._QUANTILES = analysis.compute_stats()
        self._TRANSFORM = [True, True, True, False]

        # Save parameters in class.
        self.save_hyperparameters()
        self.n_channels = window_size * n_bands

        # U-Net Architecture.
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _input_process(self, input_images):
        input_images = torch.clamp(input_images, -1e6, self.hparams.clip_value)     # clip values.
        input_images = torch.nan_to_num(input_images, nan=0)    # replace NaN with zero.

        # compute log values.
        for i, band in enumerate(self.dataset.all_bands):
            if self._TRANSFORM[i]:
                input_images[:, :, i, :, :] = torch.log(input_images[:, :, i, :, :] + 1)

        # standardize input values.
        mean = torch.tensor(self._MEAN)[None, None, :, None, None]
        mean = mean.to(input_images)    # move to device.
        std = torch.tensor(self._STD)[None, None, :, None, None]
        std = std.to(input_images)      # move to device.
        input_images = (input_images - mean) / std

        # collapse the number of bands and time steps into one dimension.
        input_images = torch.flatten(input_images, start_dim=1, end_dim=2)

        return input_images

    def _label_process(self, labeled_images):
        labeled_images = torch.log(labeled_images + 1)  # transform to log scale.
        labeled_images = (labeled_images - self._MEAN[0]) / self._STD[0]    # standardize values.
        return labeled_images

    def _reverse_preprocess(self, data):
        data = data * self._STD[0] + self._MEAN[0]
        data = torch.exp(data) - 1
        return data

    def _fixed_width_binning(self, labeled_images):
        boundaries = torch.tensor(self.hparams.bins).to(labeled_images)

        # replace NaN values with a negative value.
        labeled_images = torch.nan_to_num(labeled_images, nan=-1)

        bin_stats = dict()
        # compute mean and median for each bin.
        for i in range(self.hparams.n_classes):
            bin_vals = labeled_images[(torch.bucketize(labeled_images, boundaries) == i)]
            bin_stats[i] = {'mean': bin_vals.mean(), 'median': bin_vals.median()}

        # create class labels containing the range of values.
        boundary_vals = self.hparams.bins
        boundary_labels = []
        for i in range(len(boundaries) - 1):
            boundary_labels.append(f"{boundary_vals[i]}-{boundary_vals[i + 1]}")
        boundary_labels.append(f">{boundary_vals[-1]}")

        classification_info = {
            'boundary_labels': boundary_labels,
            'bin_stats': bin_stats
        }

        # bin labels based on boundaries.
        return torch.bucketize(labeled_images, boundaries), classification_info

    def _adaptive_binning(self, labeled_images):
        # get ranges for adaptive binning.
        boundaries = self._QUANTILES[0].to(labeled_images)

        # replace NaN values with a negative value.
        labeled_images = torch.nan_to_num(labeled_images, nan=-1)

        bin_stats = dict()
        # compute mean and median for each bin.
        for i in range(len(boundaries)):
            bin_vals = labeled_images[(torch.bucketize(labeled_images, boundaries) == i)]
            bin_stats[i] = {'mean': bin_vals.mean(), 'median': bin_vals.median()}

        # get boundary labels
        boundaries_arr = boundaries.cpu().numpy()
        boundary_labels = []
        for i in range(len(boundaries)-1):
            boundary_labels.append(f"{boundaries_arr[i]:.1f}-{boundaries_arr[i+1]:.1f}")
        boundary_labels.append(f">{boundaries_arr[-1]:.1f}")

        classification_info = {'boundary_labels': boundary_labels, 'bin_stats': bin_stats}
        return torch.bucketize(labeled_images, boundaries), classification_info

    def _process_batch(self, batch):
        input_images, _, labeled_images = batch

        # process the input.
        input_images = self._input_process(input_images)

        # mask for missing values.
        mask = torch.isnan(labeled_images)

        # clip label values.
        labeled_images = torch.clamp(labeled_images, 0, self.hparams.clip_value)

        if self.hparams.n_classes > 1:
            binned_labeled_images, classification_info = self._fixed_width_binning(labeled_images)
            # binned_labeled_images, classification_info = self._adaptive_binning(labeled_images)
            classification_info['regression_label'] = labeled_images.squeeze()
            return input_images, binned_labeled_images, mask, classification_info
        else:
            labeled_images = self._label_process(labeled_images)
            return input_images, labeled_images, mask, {}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=len(self.train_dataloader()))
        return [optimizer], [scheduler] 

    def train_dataloader(self):
        # Train split. This will sample random bounding boxes from the dataset of size input_size.
        train_roi = BoundingBox(
            minx=self.dataset.roi.minx,
            maxx=self.dataset.roi.maxx,
            miny=self.dataset.roi.miny,
            maxy=self.dataset.roi.maxy,
            mint=self.dataset.roi.mint,
            maxt=datetime(2021, 12, 31).timestamp(),
        )

        train_sampler = RandomGeoSampler(
            self.dataset.data_bio_unprocessed,
            size=float(self.dataset.input_size),
            length=self.hparams.train_samples,  # Number of iterations in one epoch.
            roi=train_roi,
        )

        train_loader = DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=train_sampler,
        )

        # overfit for debugging purposes.
        if self.hparams.overfit:
            overfit_dataset = rionegrodata.OverfitDataset(train_loader)
            return DataLoader(overfit_dataset, collate_fn=lambda x: x[0])

        return train_loader

    def _evaluation_dataloader(self):
        # Test split. This will the original images without cropping from the dataset.
        test_roi = BoundingBox(
            minx=self.dataset.roi.minx,
            maxx=self.dataset.roi.maxx,
            miny=self.dataset.roi.miny,
            maxy=self.dataset.roi.maxy,
            mint=datetime(2021, 12, 31).timestamp(),
            maxt=self.dataset.roi.maxt,
        )

        test_sampler = GridGeoSampler(
            self.dataset.data_bio_unprocessed,
            size=(self.dataset.roi.maxy - self.dataset.roi.miny, self.dataset.roi.maxx - self.dataset.roi.minx),
            stride=1,
            roi=test_roi,
            units=Units.CRS,
        )

        test_loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            sampler=test_sampler
        )

        return test_loader

    def val_dataloader(self):
        return self._evaluation_dataloader()

    def predict_dataloader(self):
        return self._evaluation_dataloader()