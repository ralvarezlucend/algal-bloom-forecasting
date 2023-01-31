import importlib
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import torch
import seaborn as sns

rionegrodata = importlib.import_module("brp-algal-bloom-forecasting.rionegrodata")


class DataAnalysis:
    def __init__(self, rionegro_dst, clip_value):
        self.clip_value = clip_value
        self.means = []
        self.stds = []
        self.quantiles = []

        # get individual datasets.
        self.data_bio = rionegro_dst.data_bio_unprocessed
        self.data_water_temp = rionegro_dst.dataset.datasets[1]
        # self.data_water_temp = rionegro_dst.dataset.datasets[0].datasets[1]
        # self.data_meteo = rionegro_dst.dataset.datasets[1]

    def compute_stats(self):
        self._compute_stats_helper(dataset=self.data_bio, transform=[True, True, True])
        self._compute_stats_helper(dataset=self.data_water_temp, transform=[False])
        # self._compute_stats_helper(self.data_meteo)
        return self.means, self.stds, self.quantiles

    def get_labels(self):
        return self._get_chlorophyll_data(self.data_bio)

    def _get_chlorophyll_data(self, dataset):
        files = [item.object for item in dataset.index.intersection(dataset.index.bounds, objects=True)]
        data = [rasterio.open(file).read() for file in files]
        data = np.stack(data, axis=1)   # join a sequence of arrays along a new axis.
        data = np.clip(data, -1e6, self.clip_value)
        return data[0]

    def _compute_stats_helper(self, dataset, transform):
        files = [item.object for item in dataset.index.intersection(dataset.index.bounds, objects=True)]
        data = [rasterio.open(file).read() for file in files]
        data = np.stack(data, axis=1)   # join a sequence of arrays along a new axis.

        # clip and transform values to log scale.
        data = np.clip(data, -1e6, self.clip_value)

        # compute quantiles per band.
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        q = torch.tensor(quantiles).double()
        for i, band in enumerate(dataset.all_bands):
            band_vals = torch.tensor(data[i, :, :, :]).double()
            band_vals = band_vals[~band_vals.isnan()]
            band_quantiles = torch.quantile(band_vals, q, dim=0)[:-1]
            self.quantiles.append(band_quantiles)

        for i in range(len(transform)):
            if transform[i]:
                data[i] = np.log(data[i] + 1)

        # compute mean and std per band.
        for i, band in enumerate(dataset.all_bands):
            self.means.append(round(np.nanmean(data[i]), 4))
            self.stds.append(round(np.nanstd(data[i]), 4))


if __name__ == "__main__":
    rionegro_dataset = rionegrodata.RioNegroData(
        root='/Users/rodrigoalvarezlucendo/Desktop/algal-bloom/data',
        reservoir='palmar',
        window_size=1,
        prediction_horizon=1,
    )

    analysis = DataAnalysis(rionegro_dataset, 250)
    means, stds, quantiles = analysis.compute_stats()

    # plot label distribution.
    # quantiles = quantiles[0]
    # fixed_ranges = [0, 10, 30, 75]
    # labels = analysis.get_labels().flatten()
    #
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 6))
    # plt.suptitle('Binning Strategies Against the Chlorophyll-a Concentration Distribution', fontsize=16)
    # sns.histplot(labels, ax=ax1)
    # sns.histplot(labels, ax=ax2)
    # ax1.set_title('Fixed-Width Binning', fontsize=14)
    # ax2.set_title('Quantile-Based Adaptive Binning', fontsize=14)
    # ax1.set_xlabel('Chlorophyll-a Concentration (ug/L)', fontsize=14)
    # ax1.set_ylabel('Frequency', fontsize=14)
    # ax2.set_xlabel('Chlorophyll-a Concentration (ug/L)', fontsize=14)
    # ax2.set_ylabel('Frequency', fontsize=14)
    #
    # # plot bin separations.
    # for i in range(len(fixed_ranges)):
    #     sep1 = ax1.axvline(fixed_ranges[i], color='r')
    #     sep2 = ax2.axvline(quantiles[i], color='r')
    #
    # ax1.legend([sep1], ['Pre-Fixed Ranges'])
    # ax2.legend([sep2], ['4-Quantiles'])
    #
    # plt.tight_layout()
    # # plt.show()
    # fig.savefig('binning_strategies.svg')

