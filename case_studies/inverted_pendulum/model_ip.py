import matplotlib.pyplot as plt
import numpy as np
import torch

from moldy.model.Model import Model


class InvertedPendulum(Model):
    def __init__(self, 
                 length:float=1.0, #1.5
                 mass:float=0.2, #0.5
                 damping:float=0.1, #0.05 
                 gravity:float=9.81, 
                 uMax:np.ndarray=np.ones((1,1)), 
                 **kwargs:dict):
        """
        :param length: float, length of the pendulum
        :param mass: float, mass of the pendulum
        :param damping: float, damping of the pendulum
        :param gravity: float, gravity
        :param uMax: np.ndarray, maximum torque, size (1,num_inputs)
        :param kwargs: dict, additional arguments
        """
        super().__init__(**kwargs)
        self.name = "inverted_pendulum"
        self.numStates = 2
        self.numInputs = 1

        self.uMax = uMax
        self.uMin = -uMax
        self.xMax = np.array([10.0*np.pi, 2.0*np.pi]).reshape([1,self.numStates])
        self.xMin = -self.xMax

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.I = self.m * self.l**2.0

    def calc_state_derivs(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        """
        Calculate the state derivatives for the inverted pendulum
        :param x: state array, np.ndarray of size (num_data_points, 2)
        :param u: command array, np.ndarray of size (num_data_points, 1)
        :return: state derivative array, np.ndarray of size (num_data_points, 2)
        """
        xdot = np.zeros(x.shape)
        xdot[:,0] = (-self.b * x[:,0] + self.m * self.g * np.sin(x[:,1]) + u.flatten()) / self.I
        xdot[:,1] = x[:,0]
        return xdot
    
    # def forward_simulate_dt(self, x: np.ndarray, u: np.ndarray, dt: float = 0.01) -> np.ndarray:
    #     wrapAngle = 1
    #     wrapRange = (-np.pi, np.pi)
    #     low = wrapRange[0]
    #     high = wrapRange[1]
    #     cycle = high - low
    #     x[:, wrapAngle] = (x[:, wrapAngle] + cycle / 2) % cycle + low

    #     return super().forward_simulate_dt(x, u, dt, method)
    
    def generate_random_state(self, initial_conditions:bool=False, n:int=1) -> np.ndarray:
        """
        Generate random state array
        :param initial_conditions: bool, if True, generate initial conditions, else generate random states
        :param n: int, number of states to generate
        :return: np.ndarray of size (n, num_states)
        """
        if type(self.xMin) == torch.Tensor:
            on_gpu = True
            self.xMin = self.xMin.cpu()
            self.xMax = self.xMax.cpu()
        else:
            on_gpu = False

        if initial_conditions:
            state = np.tile(np.array([0, -np.pi]).reshape(1,self.numStates), (n,1))
        else:
            state = super().generate_random_state(n=n)

        if on_gpu:
            self.xMin = self.xMin.cuda()
            self.xMax = self.xMax.cuda()
        return state
    
    def generate_random_xgoal(self, n:int=1) -> np.ndarray:
        """
        Generate random state goal array
        :param n: int, number of states to generate
        :return: np.ndarray of size (n, num_states)
        """
        if type(self.xMin) == torch.Tensor:
            on_gpu = True
            self.xMin = self.xMin.cpu()
            self.xMax = self.xMax.cpu()
        else:
            on_gpu = False

        state = np.zeros((n,self.numStates))
        if on_gpu:
            self.xMin = self.xMin.cuda()
            self.xMax = self.xMax.cuda()
        return state

    def visualize(self, x:np.ndarray, u:np.ndarray=np.zeros((1,1))) -> None:
        """
        Visualize the inverted pendulum
        :param x: state array, np.ndarray of size (num_data_points, 2)
        :param u: command array, np.ndarray of size (num_data_points, 1)
        """
        if type(x) == torch.Tensor:
            x = x.detach().cpu().numpy()
            u = u.detach().cpu().numpy()

        CoM = [-0.5 * np.sin(x[0,1]), 0.5 * np.cos(x[0,1])]
        theta = x[0,1]
        x = [CoM[0] + self.l / 2.0 * np.sin(theta), CoM[0] - self.l / 2.0 * np.sin(theta)]
        y = [CoM[1] - self.l / 2.0 * np.cos(theta), CoM[1] + self.l / 2.0 * np.cos(theta)]

        massX = CoM[0] - self.l / 2.0 * np.sin(theta)
        massY = CoM[1] + self.l / 2.0 * np.cos(theta)

        plt.figure(f"{self.name} Visualization")
        plt.clf()
        ax = plt.gca()
        ax.set_aspect("equal")
        plt.plot(x, y)
        plt.scatter(massX, massY, 50, "r")
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.ion()
        plt.show()
        plt.pause(1e-7)

    def visualize_horizon(self, x_history:np.ndarray, xgoal:np.ndarray, path:np.ndarray, horizon:int, sim_length:float, i:int) -> None:
        """
        Visualize the horizon of the inverted pendulum
        :param x_history: state history array, np.ndarray of size (num_data_points, num_states)
        :param xgoal: state goal array, np.ndarray of size (num_data_points, num_states)
        :param path: planned state array, np.ndarray of size (num_data_points, num_states)
        :param horizon: int, length of the horizon
        :param sim_length: float, length of the simulation
        :param i: int, current timestep
        :return: None
        """
        index = list(range(0, sim_length + horizon))

        plt.figure(f"{self.name} Horizon")
        plt.clf()
        plt.axhline(y=xgoal[1], color="k", linestyle="-")
        plt.plot(index[i : i + horizon], path[:, 1], ":b")
        plt.plot(index[:i], x_history[:i, 1], "-c")
        plt.legend(["Goal Theta", "Planned Theta", "Sim Theta"])
        plt.grid(True)
        plt.show()
        plt.pause(1e-7)



if __name__ == "__main__":
    sys = InvertedPendulum(length=1.5, mass=0.5, damping=0.05)

    x = np.zeros([1,sys.numStates])
    x[:,1] = 0.001
    u = np.zeros([1,sys.numInputs])

    dt = 0.01
    sim_time = 10
    horizon = int(sim_time / dt)

    x_history = np.zeros((horizon, sys.numStates))
    u_history = np.zeros((horizon, sys.numInputs))

    for i in range(0, horizon):
        x = sys.forward_simulate_dt(x, u, dt)

        x_history[i, :] = x.flatten()
        u_history[i, :] = u.flatten()

        # if i % 5 == 0:
        #     sys.visualize(x, u)

    sys.plot_history(x_history, u_history, xgoal=np.zeros((1,2)), block=True)
