import os
import subprocess
import wandb.sdk.service.service

from startup import Overfit, TrainClassification, TrainRegression, PredictClassification, PredictRegression, Tune

# Increase the timeout duration of the _wait_for_ports function from 30 seconds to 300 seconds.
# This patch fixes wandb failing to find ports on a slow cluster.
if "SLURM_JOB_ID" in os.environ:
    def _wait_for_ports_decorator(original_method):
        def _wait_for_ports(self, fname: str, proc: subprocess.Popen = None) -> bool:
            return any(original_method(self, fname, proc) for _ in range(10))
        return _wait_for_ports

    wandb.sdk.service.service._Service._wait_for_ports = \
        _wait_for_ports_decorator(wandb.sdk.service.service._Service._wait_for_ports)

if __name__ == "__main__":
    general_params = {
        'root': '/scratch/ralvarezlucend/algal-bloom/data',
        'reservoir': 'palmar',
        'window_size': 1,
        'prediction_horizon': 1,
        'n_bands': 4,
        'train_samples': 200,
        'batch_size': 4,
        'config': {'lr': 1e-4},
        'overfit': False,
        'clip_value': 150
    }

    classification_params = {
        'bins': [0, 10, 30, 75],
        'gamma': 5,  # parameter controlling focal loss.
        'beta': 0.99,  # weights customarily set as one of 0.9, 0.99, 0.999, 0.9999.
    }

    params = {**general_params, **classification_params}

    # Classification Setting.
    train = TrainClassification(accelerator='gpu', devices=1, max_epochs=15, params=params)
    unet_classification = train.run()
    PredictClassification(unet_classification).run()

    # Regression Setting.
    # train_regression = TrainRegression(accelerator='gpu', devices=1, max_epochs=50, params=params)
    # unet_regression = train_regression.run()
    # PredictRegression(unet_regression).run()

    # Overfit(accelerator='cpu', devices=1, max_epochs=20, params=params).run()
    # tune = Tune(accelerator='cpu', devices=1, max_epochs=15, params=params)
    # tune.run(num_samples=2)

