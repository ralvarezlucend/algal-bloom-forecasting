import importlib
from typing import Dict
from abc import ABC, abstractmethod

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from utils.plots import ClassificationPlot, RegressionPlot

unet_class = importlib.import_module("brp-algal-bloom-forecasting.ralvarezlucend.unet.unet_classification")
unet_reg = importlib.import_module("brp-algal-bloom-forecasting.ralvarezlucend.unet.unet_regression")


class Experiment(ABC):
    def __init__(self, accelerator: str, devices: int, max_epochs: int, params: Dict):
        logger = WandbLogger(project="unet", log_model=True)
        # logger = TensorBoardLogger("logs")

        self.accelerator = accelerator
        self.devices = devices
        self.max_epochs = max_epochs
        self.logger = logger

        self.params = params
        self.params['num_workers'] = 8
        self.params['bilinear'] = False

    @abstractmethod
    def run(self):
        pass


class TrainClassification(Experiment):
    """Class to train unet model in the classification setting"""

    def run(self):
        # create trainer.
        trainer = Trainer(accelerator=self.accelerator, devices=self.devices,
                          max_epochs=self.max_epochs, gradient_clip_val=1.0,
                          logger=self.logger)

        self.params['n_classes'] = len(self.params['bins']) + 1
        unet_classification = unet_class.UNetClassification(**self.params).double()
        trainer.fit(unet_classification)
        return unet_classification


class TrainRegression(Experiment):
    """Class to train unet model in the regression setting"""

    def run(self):
        # create trainer.
        trainer = Trainer(accelerator=self.accelerator, devices=self.devices,
                          max_epochs=self.max_epochs, gradient_clip_val=1.0,
                          logger=self.logger)

        self.params['n_classes'] = 1
        unet_regression = unet_reg.UNetRegression(**self.params).double()
        trainer.fit(unet_regression)
        return unet_regression


class PredictClassification:
    def __init__(self, model):
        self.model = model

    def run(self):
        # create trainer.
        trainer = Trainer(accelerator='cpu', devices=1)

        # get predictions.
        predictions = trainer.predict(self.model)
        no_predictions = trainer.num_predict_batches[0]

        # plot predictions.
        acc, mean_loss, median_loss, _ = ClassificationPlot(predictions, no_predictions).plot()
        print('AVERAGE VALIDATION ACCURACY: ', acc)
        print('MEAN VALIDATION LOSS: ', mean_loss)
        print('MEDIAN VALIDATION LOSS: ', median_loss)


class PredictRegression:
    def __init__(self, model):
        self.model = model

    def run(self):
        # create trainer.
        trainer = Trainer(accelerator='cpu', devices=1)

        # get predictions.
        predictions = trainer.predict(self.model)
        no_predictions = trainer.num_predict_batches[0]

        # plot predictions.
        acc, _, _, loss = RegressionPlot(predictions, no_predictions).plot()
        print('AVERAGE VALIDATION ACCURACY: ', acc)
        print('AVERAGE MSE LOSS: ', loss)


class Overfit(Experiment):
    """Class to overfit in one batch"""

    def run(self):
        # create overfit trainer.
        trainer = Trainer(accelerator=self.accelerator, devices=self.devices,
                          max_epochs=self.max_epochs, logger=self.logger,
                          deterministic=True, overfit_batches=1)

        # add extra arguments to params.
        self.params['overfit'] = True
        self.params['train_samples'] = 1
        self.params['batch_size'] = 1

        unet_classification = unet_class.UNetClassification(**self.params).double()
        trainer.fit(unet_classification)


class Tune(Experiment):
    """Class to tune hyper-parameters"""

    def train_tune(self, config):
        # define ray tune callback.
        metrics = {'loss': 'avg/val_loss', 'acc': 'avg/val_acc'}
        callbacks = [TuneReportCallback(metrics, on='validation_end')]

        # create trainer.
        trainer = Trainer(accelerator=self.accelerator, devices=self.devices,
                          max_epochs=self.max_epochs, logger=self.logger,
                          callbacks=callbacks)

        del self.params['config']

        unet_classification = unet_class.UNetClassification(**self.params, config=config).double()
        trainer.fit(unet_classification)

    def run(self, num_samples):
        config = {'lr': tune.loguniform(1e-4, 1e-1)}
        analysis = tune.run(
            self.train_tune,
            config=config,
            num_samples=num_samples
        )

        print(analysis.best_config)

