import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List

from .utils import rk4

class Model:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = None
        self.numStates = None
        self.numInputs = None
        self.uMax = None
        self.uMin = None
        self.xMax = None
        self.xMin = None

    def forward_simulate_dt(self, 
                            x:np.ndarray, 
                            u:np.ndarray, 
                            dt:float=0.01,
                            method:str="RK4") -> np.ndarray:
        """
        Simulate the system forward dt seconds
        :param x: state array, np.ndarray of size (num_data_points, num_states)
        :param u: command array, np.ndarray of size (num_data_points, num_inputs)
        :param dt: float, time to simulate forward
        :param method: str, integration method to use. Either "euler" or "RK4"
        :return: state array, np.ndarray of size (num_data_points, num_states)
        """
        x = deepcopy(x)
        u = deepcopy(u)

        # x = np.clip(x, self.xMin, self.xMax)
        # u = np.clip(u, self.uMin, self.uMax)

        if method == "euler":
            x_dot = self.calc_state_derivs(x, u)
            x += x_dot * dt
        elif method == "RK4":
            x += rk4(self.calc_state_derivs, x, u, dt)
        else:
            raise ValueError("Invalid integration method")

        return x

    def calc_state_derivs(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        """
        Calculate the state derivatives
        :param x: state array, np.ndarray of size (num_data_points, num_states)
        :param u: command array, np.ndarray of size (num_data_points, num_inputs)
        :return: state derivative array, np.ndarray of size (num_data_points, num_states)
        """
        raise NotImplementedError("calc_state_derivs not implemented")
    
    def generate_random_state(self, n:int=1) -> np.ndarray:
        """
        Generate random state array
        :param n: int, number of states to generate
        :return: np.ndarray of size (n, num_states)
        """
        return ((self.xMax - self.xMin) * np.random.rand(n,self.numStates) + self.xMin)

    def generate_random_xgoal(self, n:int=1) -> np.ndarray:
        """
        Generate random state goal array
        :param n: int, number of states to generate
        :return: np.ndarray of size (n, num_states)
        """
        return self.generate_random_state(n=n)

    def generate_random_command(self, n:int=1) -> np.ndarray:
        """
        Generate random command array
        :param n: int, number of commands to generate
        :return: np.ndarray of size (n, num_inputs)
        """
        if type(self.uMin) == torch.Tensor:
            return ((self.uMax - self.uMin) * torch.rand(n,self.numInputs).float().cuda() + self.uMin).cpu().numpy()
        return ((self.uMax - self.uMin) * np.random.rand(n,self.numInputs) + self.uMin)

    def visualize(self, x:np.ndarray) -> None:
        """
        Visualize the state
        :param x: state array, np.ndarray of size (num_data_points, num_states)
        """
        raise NotImplementedError("visualize not implemented")
    
    def plot_history(self, x_history:np.ndarray, u_history:np.ndarray, xgoal:np.ndarray, x_labels:List=None, u_labels:List=None, block:bool=False, save_path:str=None) -> None:
        """
        Plot the history of the system
        :param x_history: state history array, np.ndarray of size (num_data_points, num_states)
        :param u_history: command history array, np.ndarray of size (num_data_points, num_inputs)
        :param xgoal: goal state array, np.ndarray of size (num_data_points, num_states)
        :param x_labels: list of strings, labels for the state variables
        :param u_labels: list of strings, labels for the command variables
        :param block: bool, if True, block the plot
        """
        fig, axes = plt.subplots(self.numStates, 1, sharex=True, figsize=(12, 1.25*self.numStates))
        for i in range(self.numStates):
            axes[i].plot(xgoal[:, i], "-r", label='Truth Data') #'Goal State')
            axes[i].plot(x_history[:, i], ":b", label='Sim Data') #'State History')
            if x_labels is None:
                x_labels = [f"x{i}" for i in range(self.numStates)]
            axes[i].set_ylabel(x_labels[i])
            axes[i].grid(True)
        axes[-1].legend()
        fig.suptitle(f"{self.name} Model State")
        plt.xlabel('Time')

        if save_path is not None:
            plt.savefig(save_path)

        fig, axes = plt.subplots(self.numInputs, 1, sharex=True, figsize=(12, 1.5*self.numInputs))
        if self.numInputs > 1:
            for i in range(self.numInputs):
                axes[i].plot(u_history[:, i])
                if u_labels is None:
                    u_labels = [f"u{i}" for i in range(self.numInputs)]
                axes[i].set_ylabel(u_labels[i])
                axes[i].grid(True)
        else:
            axes.plot(u_history)
            axes.set_ylabel(u_labels)
            axes.grid(True)

        fig.suptitle(f"{self.name} Model Commands")
        plt.xlabel('Time')
        plt.show(block=block)