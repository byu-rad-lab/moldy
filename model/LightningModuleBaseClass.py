import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import Tuple
import glob

from .NN_Architectures import *
from .DatasetBaseClass import DatasetBaseClass


class LightningModuleBaseClass(pl.LightningModule):
    def __init__(self, config: dict):
        """
        Base class for all Lightning Modules
        :param config: dictionary containing the configuration for the model
                - n_inputs: int, number of inputs to the neural network
                - n_outputs: int, number of outputs from the neural network
                - n_hlay: int, number of hidden layers in the neural network
                - hdim: int, number of neurons in each hidden layer
                - act_fn: str, activation function for the neural network
                - loss_fn: str, loss function for the neural network
                - opt: str, optimizer for the neural network
                - lr: float, learning rate for the neural network
                - lr_schedule: str, learning rate scheduler for the neural network
                - b_size: int, batch size for the neural network training
                - num_workers: int, number of workers for the dataloader
                - nn_arch: str, neural network architecture
                - initialization_scheme: str, initialization scheme for the neural network
        """
        super().__init__()
        self.config = config
        self.dataset = DatasetBaseClass

        self.model = self.configure_model()
        self.configure_loss_function()
        config["already_trained"] = True
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        :param x: input tensor of size (batch_size, n_inputs)
        :return: output tensor of size (batch_size, n_outputs)
        """
        try:
            if x.dtype != self.model[0][0].weight.dtype:
                x = x.to(self.model[0][0].weight.dtype)
        except:
            pass
        # if self.config["nn_arch"] == "simple_fnn":
        #     x = x.squeeze(1)
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """
        Calculates training loss for a given batch
        :param batch: tuple of (input, output) tensors where input is size (batch_size, n_inputs) and output is size (batch_size, n_outputs)
        :param batch_idx: index of the batch
        :return: training loss
        """
        x, y = batch
        y_hat = self(x)

        if self.state_weights is not None:
            y = y * self.state_weights.to(y.device)
            y_hat = y_hat * self.state_weights.to(y_hat.device)

        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        train_loss = self.loss(y_hat, y).float()
        return train_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> float:
        """
        Calculates validation loss for a given batch
        :param batch: tuple of (input, output) tensors where input is size (batch_size, n_inputs) and output is size (batch_size, n_outputs)
        :param batch_idx: index of the batch
        :return: validation loss

        This function also "logs" the validation loss for use when optimizing
        """
        x, y = batch
        y_hat = self(x)

        if self.state_weights is not None:
            y = y * self.state_weights.to(y.device)
            y_hat = y_hat * self.state_weights.to(y_hat.device)

        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        val_loss = self.loss(y_hat, y).float()
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler
        :return: dictionary containing the optimizer and learning rate scheduler
        """
        if self.config["opt"] == "adam":
            optimizer = torch.optim.Adam
        elif self.config["opt"] == "sgd":
            optimizer = torch.optim.SGD
        elif self.config["opt"] == "ada":
            optimizer = torch.optim.Adagrad
        elif self.config["opt"] == "lbfgs":
            optimizer = torch.optim.LBFGS
        elif self.config["opt"] == "rmsprop":
            optimizer = torch.optim.RMSprop
        else:
            raise ValueError(
                f"Incorrect Optimizer in tune search space. Got {self.config['opt']}"
            )

        optimizer = optimizer(self.parameters(), lr=self.config["lr"], weight_decay=self.config.get("weight_decay", 0.0))

        if self.config["lr_schedule"] == "constant":
            scheduler = lr_scheduler.ConstantLR(
                optimizer, total_iters=int(self.config["max_epochs"] * 0.8)
            )
        elif self.config["lr_schedule"] == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer, total_iters=int(self.config["max_epochs"] * 0.8)
            )
        elif self.config["lr_schedule"] == "cyclic":
            scheduler = lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.config["lr"],
                max_lr=self.config["lr"] * 10,
                cycle_momentum=False,
            )
        elif self.config["lr_schedule"] == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            )
        else:
            raise ValueError(
                f"Incorrect Learning Rate Scheduler in tune search space. Got {self.config['lr_schedule']}"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def train_dataloader(self) -> DataLoader:
        """
        Creates the training dataloader
        :return: training dataloader
        """
        self.train_dataset = self.dataset(self.config, self.system, validation=False)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("b_size", 512),
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Creates the validation dataloader
        :return: validation dataloader
        """
        self.val_dataset = self.dataset(self.config, self.system, validation=True)
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get("b_size", 512),
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True,
        )
        return val_loader
    
    def configure_model(self) -> nn.Module:
        """
        Sets up the model according to the config file passed into the class
        return: nn.Module, the model
        """

        initialization_scheme = self.config.get("initialization_scheme", "xavier_uniform")
        act_fn = self.configure_activation_function()

        if self.config["nn_arch"] == "simple_fnn":
            model = SimpleLinearNN(
                                    self.config["n_inputs"],
                                    self.config["n_outputs"],
                                    self.config["n_hlay"],
                                    self.config["hdim"],
                                    act_fn,
                                    initialization_scheme,
                                    dropout=self.config.get("dropout", 0.0),
                                    )
        elif self.config["nn_arch"] == "unet":
            model = UNet(
                            self.config["n_inputs"],
                            self.config["n_outputs"],
                            act_fn,
                            initialization_scheme,
                            dropout=self.config.get("dropout", 0.0),
                            )
        elif self.config["nn_arch"] == "lstm":
            model = LSTMPredictor(
                            self.config["n_inputs"],
                            self.config["n_outputs"],
                            self.config["hdim"],
                            self.config["n_hlay"],
                            initialization_scheme,
                            )
        else:
            raise ValueError(f"Unknown Network Architecture. Got {self.config['nn_arch']}")

        if self.config.get("pretrained_model_path", None) is not None and not self.config.get("already_trained", False): 
            try:
                paths = [f"{self.config['pretrained_model_path']}/checkpoints/", f"{self.config['pretrained_model_path']}/lightning_logs/version_0/checkpoints/"]
                for path in paths:
                    model_weights = glob.glob(f"{path}/lowest_loss.ckpt")
                    if len(model_weights) > 0:
                        break

                state_dict = torch.load(model_weights[-1])["state_dict"]

                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace("model.", "")
                    new_state_dict[new_key] = value

                model.load_state_dict(new_state_dict)

                print("############################## Loaded pretrained model successfully ####################################")
            except Exception as e:
                print(f"Could not load pretrained model. Assuming that the model already has been trained. Got error: {e}")

        return model

    def configure_activation_function(self) -> nn.Module:
        """
        Configures the activation function for the model
        :return: activation function
        """
        if self.config["act_fn"] == "relu":
            act_fn = nn.ReLU()
        elif self.config["act_fn"] == "leaky_relu":
            act_fn = nn.LeakyReLU()
        elif self.config["act_fn"] == "tanh":
            act_fn = nn.Tanh()
        elif self.config["act_fn"] == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(
                f"Incorrect Activation Function in tune search space. Got {self.config['act_fn']}"
            )
        return act_fn

    def configure_loss_function(self) -> nn.Module:
        """
        Configures the loss function for the model
        :return: None
        """
        self.state_weights = self.config.get("state_weights", None)
        if self.state_weights is not None:
            self.state_weights = torch.tensor(self.state_weights).float()

        if self.config["loss_fn"] == "mse":
            self.loss = nn.MSELoss()
        elif self.config["loss_fn"] == "mae":
            self.loss = nn.L1Loss(reduction="mean")
        elif self.config["loss_fn"] == "iae":
            self.loss = nn.L1Loss(reduction="sum")
        elif self.config["loss_fn"] == "rmse":
            class RMSELoss(nn.Module):
                def __init__(self, eps=1e-6):
                    super().__init__()
                    self.mse = nn.MSELoss()
                    self.eps = eps

                def forward(self, yhat, y):
                    return torch.sqrt(self.mse(yhat, y) + self.eps)

            self.loss = RMSELoss()
        elif self.config["loss_fn"] == "cosine":
            class CosineLoss(nn.Module):
                def __init__(self):
                    super().__init__()

                def cosine_loss(self, yhat, y):
                    return torch.sum((1 - torch.cosine_similarity(yhat, y)))

                def forward(self, yhat, y):
                    return torch.linalg.norm(y - yhat, ord=1) + self.cosine_loss(
                        yhat, y
                    )

            self.loss = CosineLoss()
        elif self.config["loss_fn"] == "smooth_L1":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(
                f"Incorrect Loss Function in tune search space. Got {self.config['loss_fn']}"
            )