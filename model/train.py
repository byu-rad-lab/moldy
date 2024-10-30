from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass
from moldy.model.utils import _ModelCheckpoint, _TuneReportCallback, plot_loss_curve

def train_no_tune(
    config: dict, lighting_module_given: LightningModuleBaseClass
) -> None:
    """
    Train a model without using Ray Tune
    :param config: dictionary containing the configuration for the model
    :param lighting_module_given: LightningModuleBaseClass object
    :return: None
    """
    lightning_module = lighting_module_given(config)

    checkpoint_callback = _ModelCheckpoint(
        save_top_k=1,
        monitor=config["metric"],
        mode=config["mode"],
        filename="lowest_loss",
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        deterministic=True,
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        logger=CSVLogger(
            save_dir=f"{config['path']}results/run_logs/{config['run_name']}"
        ),
        callbacks=[checkpoint_callback],  # EarlyStopping(monitor=config["metric"], mode=config["mode"], patience=150)
    )
    
    trainer.fit(lightning_module)

    plot_loss_curve(config)


def train_tune(config: dict, lighting_module_given: LightningModuleBaseClass) -> None:
    """
    Train a model using Ray Tune optimized hyperparameters
    :param config: dictionary containing the configuration for the model
    :param lighting_module_given: LightningModuleBaseClass object
    :return: None
    """
    lightning_module = lighting_module_given(config)
    checkpoint_callback = _ModelCheckpoint(
        save_top_k=1,
        monitor=config["metric"],
        mode=config["mode"],
        filename="lowest_loss",
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        deterministic=True,
        max_epochs=config["max_epochs"],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        callbacks=[_TuneReportCallback("val_loss"), checkpoint_callback],
    )
    trainer.fit(lightning_module)