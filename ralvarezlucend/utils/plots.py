import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from matplotlib.colors import ListedColormap
import os
import numpy as np
import torch
from torchmetrics import ConfusionMatrix


class Plot(ABC):
    def __init__(self, predictions, no_predictions):
        self.predictions = predictions
        self.no_predictions = no_predictions
        self.accuracies = list()
        self.mean_losses = list()
        self.median_losses = list()
        self.losses = list()
        self.confusion_matrices = list()

    def plot(self):
        results = "test"
        # create results directory.
        if not os.path.exists(results):
            os.mkdir(results)

        for idx in range(0, self.no_predictions):
            # create subdirectory for specific prediction.
            path = os.path.join(results, f"figure_{idx}")
            if not os.path.exists(path):
                os.mkdir(path)

            # plot the prediction.
            output_image, labeled_image, mask, classification_info = self.predictions[idx]
            self.make_plot(output_image, labeled_image, mask, path, classification_info)

        # average the confusion matrices of all the samples in the validation set.
        confusion_matrix = torch.mean(torch.stack(self.confusion_matrices), dim=0,
                                      keepdim=True, dtype=torch.float32).squeeze()

        # plot confusion matrix.
        plt.figure()
        heat_map = sns.heatmap(confusion_matrix, annot=True, fmt='.2f')
        heat_map.set_xlabel('Predicted label', fontsize=16)
        heat_map.set_ylabel('True label', fontsize=16)
        heat_map.set_title('Normalized confusion matrix', fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{results}/confusion_matrix.pdf")

        return np.mean(self.accuracies), np.mean(self.mean_losses), np.mean(self.median_losses), np.mean(self.losses)

    @abstractmethod
    def make_plot(self, output_image, labeled_image, mask, path):
        pass

    @staticmethod
    def compute_accuracy(input, target, mask):
        matching_labels = (input == target)
        correct = matching_labels[~mask].sum()
        wrong = len(matching_labels[~mask]) - correct
        acc = (correct / (correct + wrong))
        return acc, matching_labels


class ClassificationPlot(Plot):

    def make_plot(self, output_image, labeled_image, mask, path, classification_info):
        fig, axs = plt.subplot_mosaic('AD;BE;CF', figsize=(6*2, 4*3))
        fig.suptitle('Classification results of forecasting chlorophyll-a concentration',
                     fontsize=18)

        background_color = '#eaeaf2'

        # shared settings.
        vmin, vmax = 1, 4
        labels = classification_info['boundary_labels']
        n = len(labels)
        r = vmax - vmin
        cmap = sns.color_palette("hls", n)

        # plot label.
        axs['A'].set_title('Label', fontsize=16)
        axs['A'].set_facecolor(background_color)

        sns.heatmap(labeled_image, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask,
                    yticklabels=False, xticklabels=False, ax=axs['A'])

        # set label colorbar.
        colorbar = axs['A'].collections[0].colorbar
        colorbar.set_ticks([vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(labels)
        colorbar.ax.tick_params(labelsize=16)

        # plot prediction.
        axs['B'].set_title('Prediction', fontsize=16)
        axs['B'].set_facecolor(background_color)

        predicted_image = output_image.argmax(dim=0)
        sns.heatmap(predicted_image, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask,
                    yticklabels=False, xticklabels=False, ax=axs['B'])

        # set prediction colorbar.
        colorbar = axs['B'].collections[0].colorbar
        colorbar.set_ticks([vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(labels)
        colorbar.ax.tick_params(labelsize=16)

        # plot accuracy.
        acc, matching_labels = self.compute_accuracy(predicted_image, labeled_image, mask)
        self.accuracies.append(acc)  # save accuracy.
        axs['C'].set_title(f'Accuracy ({(acc*100):.2f}% correct)', fontsize=16)
        axs['C'].set_facecolor(background_color)

        colors = ["#805500", "#F4B400"]
        cmap_acc = ListedColormap(sns.color_palette(colors))
        sns.heatmap(matching_labels, cmap=cmap_acc, mask=mask, yticklabels=False,
                    xticklabels=False, ax=axs['C'])

        # configuration acc.
        labels = ['Wrong', 'Correct']
        n = len(labels)
        vmin, vmax = 0, 1
        r = vmax - vmin

        # set accuracy colorbar.
        colorbar = axs['C'].collections[0].colorbar
        colorbar.set_ticks([vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(labels)
        colorbar.ax.tick_params(labelsize=16)

        # plot probabilities of the most probable class.
        axs['D'].set_title('Confidence of most probable class (%)', fontsize=16)
        axs['D'].set_facecolor(background_color)

        probs_most_probable = torch.max(output_image, 0).values * 100
        cmap_confidence = sns.cubehelix_palette(as_cmap=True)
        sns.heatmap(probs_most_probable, mask=mask, yticklabels=False,
                    xticklabels=False, cmap=cmap_confidence, ax=axs['D'])

        colorbar = axs['D'].collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)

        # plot regression conversion.
        bin_stats = classification_info['bin_stats']
        mean_smooth_image = predicted_image
        median_smooth_image = predicted_image
        for bin, bin_stat in bin_stats.items():
            mean_smooth_image = torch.where(mean_smooth_image == bin, bin_stat['mean'], mean_smooth_image)
            median_smooth_image = torch.where(median_smooth_image == bin, bin_stat['median'], median_smooth_image)

        cmap_smoothed = sns.color_palette("crest", as_cmap=True)

        axs['E'].set_title('Prediction smoothed by bin mean', fontsize=16)
        axs['E'].set_facecolor(background_color)
        sns.heatmap(mean_smooth_image, mask=mask, yticklabels=False,
                    xticklabels=False, cmap=cmap_smoothed, ax=axs['E'])

        colorbar = axs['E'].collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)

        axs['F'].set_title('Prediction smoothed by bin median', fontsize=16)
        axs['F'].set_facecolor(background_color)
        sns.heatmap(median_smooth_image, mask=mask, yticklabels=False,
                    xticklabels=False, cmap=cmap_smoothed, ax=axs['F'])

        colorbar = axs['F'].collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)

        # compute mse loss in the classification setting.
        regression_label = classification_info['regression_label']
        mean_loss = ((regression_label[~mask]-mean_smooth_image[~mask])**2).nanmean()
        median_loss = ((regression_label[~mask]-median_smooth_image[~mask])**2).nanmean()
        self.mean_losses.append(mean_loss)
        self.median_losses.append(median_loss)

        # create confusion matrix.
        confmat = ConfusionMatrix(task="multiclass", num_classes=5,
                                  normalize='true', ignore_index=0)
        matrix = confmat(predicted_image, labeled_image)
        matrix = matrix[1:, 1:]     # filter label 0.
        self.confusion_matrices.append(matrix)

        plt.tight_layout()
        plt.savefig(os.path.join(path, "classification_plots.pdf"))


class RegressionPlot(Plot):
    def make_plot(self, output_image, labeled_image, mask, path, classification_info):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6*2, 4*2))
        fig.suptitle('Regression results of forecasting chlorophyll-a concentration',
                     fontsize=18)
        cmap = sns.color_palette("crest", as_cmap=True)
        background_color = '#eaeaf2'

        # plot regression label.
        axs[0, 0].set_title('Label', fontsize=16)
        axs[0, 0].set_facecolor(background_color)
        sns.heatmap(labeled_image, mask=mask, yticklabels=False, xticklabels=False,
                    linewidths=0.0, linecolor='none', cmap=cmap, ax=axs[0, 0])

        colorbar = axs[0, 0].collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)

        # plot regression prediction.
        axs[0, 1].set_title('Prediction', fontsize=16)
        axs[0, 1].set_facecolor(background_color)
        sns.heatmap(output_image, mask=mask, yticklabels=False, xticklabels=False,
                    cmap=cmap, vmin=labeled_image.min(), vmax=labeled_image.max(),
                    linewidths=0.0, linecolor='none', ax=axs[0, 1])

        colorbar = axs[0, 1].collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)

        # plot error in prediction.
        cmap_error = sns.color_palette("flare", as_cmap=True)
        prediction_error = labeled_image - output_image
        axs[1, 0].set_title('Prediction error', fontsize=16)
        axs[1, 0].set_facecolor(background_color)
        sns.heatmap(prediction_error, mask=mask, yticklabels=False, xticklabels=False,
                    cmap=cmap_error, ax=axs[1, 0])

        colorbar = axs[1, 0].collections[0].colorbar
        colorbar.ax.tick_params(labelsize=16)

        # plot binned regression prediction.
        boundaries = torch.tensor([0, 10, 30, 75])
        binned_prediction = torch.bucketize(output_image, boundaries)
        binned_label = torch.bucketize(labeled_image, boundaries)
        acc, _ = self.compute_accuracy(binned_prediction, binned_label, mask)
        self.accuracies.append(acc)     # save accuracy.
        axs[1, 1].set_title(f"Binned prediction ({acc*100:.2f}% correct)", fontsize=16)

        vmin, vmax = 1, 4
        labels = ['0-10', '10-30', '30-75', '>75']
        n = len(labels)
        r = vmax - vmin
        # cmap_classification = sns.color_palette("hls", n)
        cmap_classification = sns.color_palette("hls", n_colors=n)
        axs[1, 1].set_facecolor(background_color)

        sns.heatmap(binned_prediction, cmap=cmap_classification, vmin=vmin, vmax=vmax,
                    mask=mask, yticklabels=False, xticklabels=False, ax=axs[1, 1])

        # set colorbar.
        colorbar = axs[1, 1].collections[0].colorbar
        colorbar.set_ticks([vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(labels)
        colorbar.ax.tick_params(labelsize=16)

        # compute mse loss.
        mse = ((output_image[~mask] - labeled_image[~mask]) ** 2).mean()
        self.losses.append(mse)

        # save figure.
        fig.tight_layout()
        plt.savefig(os.path.join(path, "regression_plots.pdf"))