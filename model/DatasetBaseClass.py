import numpy as np
import os
from torch.utils.data import Dataset
from typing import Tuple

from moldy.model.Model import Model
from .utils import rk4 
from moldy.utils.trajectory_utils import generate_square_wave


class DatasetBaseClass(Dataset):
    def __init__(self, config: dict, system: Model, validation: bool = False):
        """
        Base class for creating datasets for training and validation from a given system
        :param config: dict containing the following keys:
            "dataset_size": int, number of data points to generate
            "dt": float, time step
            "path": str, path to save data to
            "learn_mode": str, "delta_x" or "x", default is "delta_x"
            "normalization_method": str, "min_max", "standardized", "mean_std", "none", default is "mean_std"
            "generate_new_data": bool, whether or not to generate new data, default is False
        :param system: Model, system to generate data from
        :param validation: bool, true is to generate validation data, false is to generate training data
        """
        assert (
            type(system) is not None
        ), "System type is None. Check that the Lightning Module Base class is properly implemented"
        self.system = system
        self.validation = validation
        self.data_generation_params = config["data_generation_params"]
        self.learn_mode = self.data_generation_params.get("learn_mode", "delta_x")
        self.normalization_method = self.data_generation_params.get("normalization_method", "max")
        self.dt = self.data_generation_params.get("dt", 0.01)
        self.cutoff_data_amount = 5000
        self.size = int(self.data_generation_params["dataset_size"] * (0.2 if validation else 0.8))
        self.data_type = self.data_generation_params.get("type", "step")
        generate_new_data = self.data_generation_params.get("generate_new_data", False)

        try:
            os.makedirs(config["path"] + "data/")
        except:
            pass
        self.path = config["path"] + (
            "data/validation_" if validation else "data/train_"
        )

        self.generate_data_if_needed(generate_new_data)

    def generate_data_if_needed(self, generate_new_data: bool) -> None:
        """
        Generate data if needed
        :param generate_new_data: bool, whether or not to generate new data
        :return: None
        """
        if not generate_new_data:
            try:
                self.input_data = np.load(
                    self.path + "inputdata.npy"
                )
                self.output_data = np.load(
                    self.path + "outputdata.npy"
                )

                if (
                    self.input_data.shape[0] != self.output_data.shape[0]
                    or self.input_data.shape[0] != self.size
                ) and self.size != 0:
                    raise ValueError(
                        f"Size mismatch between input (size={self.input_data.shape}) and output (size={self.output_data.shape}) data. Expects data of shape ({self.size},{self.system.numStates}). Generating new data..."
                    )
            except Exception as e:
                print(f"Could not load data. Will generate new data. Got the following error: {e}")
                generate_new_data = True

        if generate_new_data:
            if self.data_type == "random" and self.size > 0:
                input_data, output_data = self.generate_random_data(self.size + self.cutoff_data_amount)
            elif self.data_type == "step" and self.size > 0:
                input_data, output_data = self.generate_step_data(self.size + self.cutoff_data_amount)
            elif self.data_type == "sine" and self.size > 0:
                input_data, output_data = self.generate_sine_data(self.size + self.cutoff_data_amount)

            self.input_data = input_data[self.cutoff_data_amount:].astype(np.float32)
            self.output_data = output_data[self.cutoff_data_amount:].astype(np.float32)
            np.save(self.path + "inputdata", self.input_data)
            np.save(self.path + "outputdata", self.output_data)

    def generate_random_data(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random data
        :param size: int, number of data points to generate
        :return: tuple of (input_data, output_data) where input_data is of size (size, num_states + num_inputs) and output_data is of size (size, num_states)
        """
        state_array = self.system.generate_random_state(n=size)
        command_array = self.system.generate_random_command(n=size)

        input_data, output_data = self.simulate_data(size, state_array, command_array)

        if self.normalization_method != "none":
            input_data, output_data = self.normalize_data(input_data, output_data)

        return input_data, output_data

    def generate_step_data(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from step commands in control input space
        :param size: int, number of data points to generate
        :return: tuple of (input_data, output_data) where input_data is of size (size, num_states + num_inputs) and output_data is of size (size, num_states)
        """
        state_array = self.system.generate_random_state(n=1)
        command_array = generate_square_wave(self.system.numInputs, size, self.system.uMin, self.system.uMax)

        input_data, output_data = self.simulate_sequential_data(size, state_array, command_array)

        if self.normalization_method != "none":
            input_data, output_data = self.normalize_data(input_data, output_data)

        return input_data, output_data

    def generate_sine_data(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO this needs to be fixed to have better sine wave generation
        Generate data from sine commands in control input space
        :param size: int, number of data points to generate
        :return: tuple of (input_data, output_data) where input_data is of size (size, num_states + num_inputs) and output_data is of size (size, num_states)
        """
        state_array = self.system.generate_random_state(n=1)
        command_array = np.zeros((size, self.system.numInputs))
        for i in range(self.system.numInputs):
            command_array[:, i] = np.sin(3 * np.linspace(0, 2 * np.pi, size)) * (self.system.uMax[i] - self.system.uMin[i]) / 2 + (self.system.uMax[i] + self.system.uMin[i]) / 2

        input_data, output_data = self.simulate_sequential_data(size, state_array, command_array)

        if self.normalization_method != "none":
            input_data, output_data = self.normalize_data(input_data, output_data)

        return input_data, output_data
    
    def simulate_data(
        self, size: int, state_array: np.ndarray, command_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate data using the system according to learn mode
        :param size: int, number of data points to generate
        :param state_array: np.ndarray, array of size (size, num_states)
        :param command_array: np.ndarray, array of size (size, num_inputs)
        :return: tuple of (input_data, output_data) where input_data is of size (size, num_states + num_inputs) and output_data is of size (size, num_states)
        """
        input_data = np.zeros((size, self.system.numStates + self.system.numInputs))
        output_data = np.zeros((size, self.system.numStates))
        for j in range(size):
            state = state_array[j].reshape(1, -1)
            command = command_array[j].reshape(1, -1)

            if self.learn_mode == "delta_x":
                next_state = rk4(self.system.calc_state_derivs, state, command, self.dt)
            elif self.learn_mode == "x":
                next_state = self.system.forward_simulate_dt(state, command, self.dt)

            input_data[j] = np.hstack((state, command)).squeeze()
            output_data[j] = next_state.squeeze()
        return input_data, output_data

    def simulate_sequential_data(self, size: int, state_array: np.ndarray, command_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate data that is sequential (datapoints depend on each other)

        :param size: int, number of data points to generate
        :param state_array: np.ndarray, array of size (1, num_states)
        :param command_array: np.ndarray, array of size (size, num_inputs)
        :return: tuple of (input_data, output_data) where input_data is of size (size, num_states + num_inputs) and output_data is of size (size, num_states)
        """
        input_data = np.zeros((size+1, self.system.numStates + self.system.numInputs))
        output_data = np.zeros((size+1, self.system.numStates))

        mujoco_flag = False

        for i in range(size+1):
            j = i if i < size else -1
            command = command_array[j].reshape(1, -1)

            if mujoco_flag:
                next_state = self.system.forward_simulate_dt(state_array, command, self.dt)
                next_state -= state_array
            elif self.learn_mode == "delta_x" and not mujoco_flag:
                try:
                    next_state = rk4(self.system.calc_state_derivs, state_array, command, self.dt)
                except:
                    mujoco_flag = True
                    next_state = self.system.forward_simulate_dt(state_array, command, self.dt)
                    next_state -= state_array
            elif self.learn_mode == "x":
                next_state = self.system.forward_simulate_dt(state_array, command, self.dt)
    
            input_data[i] = np.hstack((state_array, command)).squeeze()
            output_data[i] = next_state.squeeze()
            
            if self.learn_mode == "delta_x":
                state_array += next_state
                state_array = np.clip(state_array, self.system.xMin, self.system.xMax)
            elif self.learn_mode == "x":
                state_array = next_state

        input_data = input_data[1:]
        output_data = output_data[1:]

        return input_data, output_data
    
    def normalize_data(self, input_data: np.ndarray, output_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes data according to user specified options
        :param input_data: np.ndarray, input data of size (num_data_points, num_states + num_inputs)
        :param output_data: np.ndarray, output data of size (num_data_points, num_states)
        :return: tuple of (input_data, output_data) where input_data is of size (num_data_points, num_states + num_inputs) and output_data is of size (num_data_points, num_states)
        """

        if self.normalization_method == "max":
            input_data[:, : self.system.numStates] = input_data[:, :self.system.numStates] / self.system.xMax
            input_data[:, self.system.numStates :] = input_data[:, self.system.numStates :] / self.system.uMax

            try:
                output_max = np.load("/".join(self.path.split("/")[:-1]) + "output_max.npy")
            except:
                output_max = np.max(output_data, axis=0)
            output_data = output_data / output_max

            np.save("/".join(self.path.split("/")[:-1]) + "/output_max", output_max)

        elif self.normalization_method == "mean_std":
            state_mean = ((self.system.xMax + self.system.xMin) / 2).flatten()
            state_std = ((self.system.xMax - self.system.xMin) / 2).flatten()
            input_mean = (self.system.uMax + self.system.uMin) / 2
            input_std = (self.system.uMax - self.system.uMin) / 2

            input_data[:, : self.system.numStates] = (input_data[:, : self.system.numStates] - state_mean) / state_std

            input_data[:, self.system.numStates :] = (input_data[:, self.system.numStates :] - input_mean) / input_std

            output_data = (output_data - state_mean) / state_std

        elif self.normalization_method == "min_max":
            if self.validation:
                state_mean = np.load("/".join(self.path.split("/")[:-1]) + "/state_mean.npy")
                state_std = np.load("/".join(self.path.split("/")[:-1]) + "/state_std.npy")
                state_max = np.load("/".join(self.path.split("/")[:-1]) + "/state_max.npy")

                input_mean = np.load("/".join(self.path.split("/")[:-1]) + "/input_mean.npy")
                input_std = np.load("/".join(self.path.split("/")[:-1]) + "/input_std.npy")
                input_max = np.load("/".join(self.path.split("/")[:-1]) + "/input_max.npy")

                output_mean = np.load("/".join(self.path.split("/")[:-1]) + "/output_mean.npy")
                output_std = np.load("/".join(self.path.split("/")[:-1]) + "/output_std.npy")
                output_max = np.load("/".join(self.path.split("/")[:-1]) + "/output_max.npy")

                scaled_state = (input_data[:, : self.system.numStates] - state_mean) / state_std
                scaled_input = (input_data[:, self.system.numStates :] - input_mean) / input_std
                scaled_output = (output_data - output_mean) / output_std
            else:
                state_mean = np.mean(input_data[:, : self.system.numStates], axis=0)
                state_std = np.std(input_data[:, : self.system.numStates], axis=0)
                input_mean = np.mean(input_data[:, self.system.numStates :], axis=0)
                input_std = np.std(input_data[:, self.system.numStates :], axis=0)
                output_mean = np.mean(output_data, axis=0)
                output_std = np.std(output_data, axis=0)

                scaled_state = (input_data[:, : self.system.numStates] - state_mean) / state_std
                scaled_input = (input_data[:, self.system.numStates :] - input_mean) / input_std
                scaled_output = (output_data - output_mean) / output_std

                state_max = np.max(np.abs(scaled_state), axis=0)
                input_max = np.max(np.abs(scaled_input), axis=0)
                output_max = np.max(np.abs(scaled_output), axis=0)

                np.save("/".join(self.path.split("/")[:-1]) + "/state_mean", state_mean)
                np.save("/".join(self.path.split("/")[:-1]) + "/state_std", state_std)
                np.save("/".join(self.path.split("/")[:-1]) + "/state_max", state_max)

                np.save("/".join(self.path.split("/")[:-1]) + "/input_mean", input_mean)
                np.save("/".join(self.path.split("/")[:-1]) + "/input_std", input_std)
                np.save("/".join(self.path.split("/")[:-1]) + "/input_max", input_max)

                np.save("/".join(self.path.split("/")[:-1]) + "/output_mean", output_mean)
                np.save("/".join(self.path.split("/")[:-1]) + "/output_std", output_std)
                np.save("/".join(self.path.split("/")[:-1]) + "/output_max", output_max)

            input_data[:, : self.system.numStates] = scaled_state / state_max
            input_data[:, self.system.numStates :] = scaled_input / input_max
            output_data = scaled_output / output_max

        else:
            raise ValueError(f"Normalization method {self.normalization_method} not recognized")

        return input_data, output_data

    def __len__(self) -> int:
        return self.input_data.shape[0]

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.input_data[idx], self.output_data[idx]
