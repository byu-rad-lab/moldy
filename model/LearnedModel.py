import torch
from typing import Tuple
import glob, csv, yaml
import numpy as np
try:
    from torch2trt import torch2trt
except:
    print("Couldn't load torch2trt. Moving on...")

from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass

class LearnedModel:
    def __init__(self, LightningModule: LightningModuleBaseClass, trial_dir:str=None, **kwargs):
        """
        :param LightningModule: LightningModuleBaseClass for loading the model
        :param trial_dir: str
        """
        self.normalization_method = kwargs.get("normalization_method", "none")
        self.learn_mode = None
        self.model = None
        self.minLoss = None
        self.isHardware = False
        if trial_dir is not None:
            self.load_model(LightningModule, trial_dir)
            self.model = self.model.cuda()
            self.model.eval()

        use_tensorrt = kwargs.get("use_tensorrt", False)
        tensor_rt_batch_size = kwargs.get("tensor_rt_batch_size", 500)
        use_gpu = kwargs.get("use_gpu", True)
        kwargs.clear()
        super().__init__(**kwargs)

        if use_tensorrt:
            self.model = self.convert_to_tensorrt(tensor_rt_batch_size)

        if "mean_std" in self.normalization_method:
            self.state_mean = ((self.xMax + self.xMin) / 2)
            self.state_std = ((self.xMax - self.xMin) / 2)
            self.input_mean = ((self.uMax + self.uMin) / 2)
            self.input_std = (self.uMax - self.uMin) / 2
        elif "min_max" in self.normalization_method:
            self.state_mean = np.load(self.path + "/data/state_mean.npy")
            self.state_std = np.load(self.path + "/data/state_std.npy")
            self.input_mean = np.load(self.path + "/data/input_mean.npy")
            self.input_std = np.load(self.path + "/data/input_std.npy")
            self.state_max = np.load(self.path + "/data/state_max.npy")
            self.input_max = np.load(self.path + "/data/input_max.npy")
            self.output_mean = np.load(self.path + "/data/output_mean.npy")
            self.output_std = np.load(self.path + "/data/output_std.npy")
            self.output_max = np.load(self.path + "/data/output_max.npy")
        elif self.normalization_method == "max":
            self.output_max = torch.from_numpy(np.load(self.path + "/data/output_max.npy").reshape(1, -1)).float().cuda()

        if use_gpu:
            self.xMin = torch.from_numpy(self.xMin).float().cuda()
            self.xMax = torch.from_numpy(self.xMax).float().cuda()
            self.uMax = torch.from_numpy(self.uMax).float().cuda()
            self.uMin = torch.from_numpy(self.uMin).float().cuda()

            if "mean_std" in self.normalization_method:
                self.state_mean = torch.from_numpy(self.state_mean).float().cuda()
                self.state_std = torch.from_numpy(self.state_std).float().cuda()
                self.input_mean = torch.from_numpy(self.input_mean).float().cuda()
                self.input_std = torch.from_numpy(self.input_std).float().cuda()

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.model(*args, **kwds)
    
    def normalize_data(self, x:torch.Tensor, u:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize the data based user specified options
        :param x: torch.Tensor, state data of size (num_data_points, num_states)
        :param u: torch.Tensor, input data of size (num_data_points, num_inputs)
        :return: Tuple[torch.Tensor, torch.Tensor], normalized state and input data of same size as input
        """
        if self.normalization_method == "max":
            xNorm = x / self.xMax #self.input_max[:, :self.numStates]
            uNorm = u / self.uMax #self.input_max[:, self.numStates:]
        elif "mean_std" in self.normalization_method:
            xNorm = ((x - self.state_mean) / self.state_std)
            uNorm = ((u - self.input_mean) / self.input_std)
        elif "min_max" in self.normalization_method:
            xNorm = ((x - self.state_mean) / self.state_std) / self.state_max
            uNorm = ((u - self.input_mean) / self.input_std) / self.input_max
        return xNorm, uNorm
    
    def denormalize_data(self, x:torch.Tensor) -> torch.Tensor:
        """
        Denormalize the data based on user specified options
        :param x: torch.Tensor, state data of size (num_data_points, num_states)
        :return: torch.Tensor, denormalized state data of same size as input
        """
        if self.normalization_method == "max":
            if self.output_max is not None:
                return x * self.output_max
            else:
                return x * self.xMax
        elif self.normalization_method == "min_max":
            return (x * self.state_std) + self.state_mean
        else:
            return ((x * self.state_std) + self.state_mean)

    def calc_state_derivs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Calculate the state derivatives for the learned model
        :param x: torch.Tensor, state data of size (num_data_points, num_states)
        :param u: torch.Tensor, input data of size (num_data_points, num_inputs)
        :return: torch.Tensor, state derivative data of size (num_data_points, num_states)
        """
        if self.normalization_method != "none":
            x, u = self.normalize_data(x, u)

        inputs = torch.cat([x, u], dim=1).float().cuda()
        xdot = self.model(inputs)

        if self.normalization_method != "none" and not "no_output" in self.normalization_method:
            xdot = self.denormalize_data(xdot)
            
        return xdot

    def forward_simulate_dt(self, x:torch.Tensor, u:torch.Tensor, dt:float=0.01) -> torch.Tensor:
        """
        Forward simulate the model using the neural network
        :param x: torch.Tensor, state data of size (num_data_points, num_states)
        :param u: torch.Tensor, input data of size (num_data_points, num_inputs)
        :param dt: float, time step
        :return: torch.Tensor, state data of size (num_data_points, num_states)

        dt is not used because the models learn a constant dt. This could be changed in the future to be able
        predict using a variable dt.
        """

        x = torch.clamp(x, self.xMin, self.xMax)
        u = torch.clamp(u, self.uMin, self.uMax)

        if self.learn_mode == "delta_x":
            delta_x = self.calc_state_derivs(x, u)
            x_next = x + delta_x
        elif self.learn_mode == "x":
            x = self.calc_state_derivs(x, u)
            raise NotImplementedError("This mode is not implemented yet")
        else:
            raise ValueError(f"Invalid learn_mode. Must be x or delta_x. Got {self.learn_mode}")
        return x_next


    def convert_to_tensorrt(self, batch_size:int=500, precision:str="fp16"):
        """
        Convert a PyTorch model to a TensorRT model
        :param precision: str, precision for the TensorRT model
        :return: TensorRT model
        """
        x = torch.ones((batch_size, self.numInputs+self.numStates)).cuda()
        model_trt = torch2trt(self.model, [x], fp16_mode=(precision == "fp16"))
        return model_trt

    def load_model(self, LightningModule: LightningModuleBaseClass, trial_dir:str) -> None:
        """
        Load the model from the trial directory
        :param LightningModule: LightningModuleBaseClass inherited for a given system for loading the model
        :param trial_dir: str, path to the trial directory
        :return: None

        This function will load the model with the lowest validation loss from training datas
        """
        checkpoint_path = glob.glob(trial_dir + "/**/lowest_loss.ckpt",recursive=True)
        if checkpoint_path == []:
            checkpoint_path = glob.glob(trial_dir + "/**/*.ckpt",recursive=True)[-1]
        else:
            checkpoint_path = checkpoint_path[-1]
        config_path = glob.glob(trial_dir + "/**/hparams.yaml",recursive=True)[-1]
        metrics_path = glob.glob(trial_dir + "/**/metrics.csv",recursive=True)
        if metrics_path == []:
            metrics_path = glob.glob(trial_dir + "/**/progress.csv",recursive=True)[-1]
        else:
            metrics_path = metrics_path[-1]

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["config"]
        
        with open(metrics_path, "r") as f:
            resultList = []
            csv_reader = csv.reader(f, delimiter=",")
            loss_idx = 0
            for row in csv_reader:
                if not row[0][0].isalpha():
                    loss = float(row[loss_idx])
                    resultList.append(loss)
                else:
                    loss_idx = row.index(next(item for item in row if 'loss' in item))

        self.minLoss = min(resultList)
        self.normalization_method = config.get("data_generation_params", {}).get("normalization_method", "max")
        self.learn_mode = config.get("learn_mode", "delta_x")
        self.path = config.get("path", "")

        self.model = LightningModule.load_from_checkpoint(checkpoint_path, config=config)

    def get_model_loss(self) -> float:
        """
        Get the model loss
        :return: float, minimum model loss obtained during training (corresponding to the loaded weights)
        """
        return self.minLoss