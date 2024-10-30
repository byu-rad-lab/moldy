import numpy as np
import matplotlib.pyplot as plt
import torch

from moldy.model.Model import Model
from moldy.case_studies.grub_sim.model import bellows_grub_dynamics as dyn

class GrubSim(Model):
    def __init__(
        self, 
        mass:float=5.0, #2.0
        stiffness:float=14.07, #36.0, 
        damping:float= 2.73, #2.25, 
        pressure_resp_coeff:float=2.856, #20.0,
        # h:float=0.21, #0.21 not letting h be set because it seems to make everything wonky.
        r:float=0.095, #0.16
        **kwargs:dict
    ):
        """
        :param mass: float, mass of the grub
        :param stiffness: float, stiffness of the grub
        :param damping: float, damping of the grub
        :param pressure_resp_coeff: float, pressure response coefficient of the grub
        :param h: float, height of the grub
        :param r: float, radius of the grub
        :param kwargs: dict, additional arguments   
        

        state is pressures and qdot and q - in kPa and rad/s and rad     
        """
        old_params = kwargs.get("old_params", False)
        vary_params_percent = kwargs.get("vary_params_percent", 0.0)
        kwargs.pop("h", None)
        kwargs.pop("old_params", None)
        kwargs.pop("vary_params_percent", None)

        super().__init__(**kwargs)
        
        self.name = "grub"
        self.numStates = 8
        self.numInputs = 4
        self.plotting = False

        self.max_pressure = 275 # params["general"]["max_pressure"] / self.mujoco_pressure_multiplier

        self.uMax = np.ones((1,self.numInputs))*self.max_pressure
        self.uMin = self.uMax*0.0
        self.xMax = np.array(
            [[
                self.max_pressure,self.max_pressure,self.max_pressure,self.max_pressure,   
                2*np.pi,2*np.pi,
                np.pi/2,np.pi/2]])
        self.xMin = np.array(
            [[
                0,0,0,0,
              -2*np.pi,-2*np.pi,
              -np.pi/2,-np.pi/2,]])

        if vary_params_percent > 0.0:
            self.stiffness = np.random.uniform(1-vary_params_percent, 1 + vary_params_percent) * stiffness
            self.damping = np.random.uniform(1-vary_params_percent, 1 + vary_params_percent) * damping
            self.alpha = np.random.uniform(1-vary_params_percent, 1 + vary_params_percent) * pressure_resp_coeff
            self.m = np.random.uniform(1-vary_params_percent, 1 + vary_params_percent) * mass
            self.h = 0.23 # np.random.uniform(1-vary_params_percent, 1 + vary_params_percent) * h #  system get wonky when h is varied
            self.r = np.random.uniform(1-vary_params_percent, 1 + vary_params_percent) * r
        elif not old_params: # system id params
            self.stiffness = stiffness
            self.damping = damping
            self.alpha = pressure_resp_coeff
            self.m = mass
            self.h = 0.23
            self.r = r
        else: # params before performing system identification
            self.stiffness = 36.0
            self.damping = 2.25
            self.alpha = 20.0
            self.m = 2.0
            self.h = 0.21
            self.r = 0.16

        print(f"GrubSim: mass: {self.m}, stiffness: {self.stiffness}, damping: {self.damping}, alpha: {self.alpha}, h: {self.h}, r: {self.r}")
    
    
    def calc_state_derivs(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        """
        Calculate the state derivatives for the grub
        :param x: state array, np.ndarray of size (num_data_points, 8)
        :param u: command array, np.ndarray of size (num_data_points, 4)
        :return: state derivative array, np.ndarray of size (num_data_points, 8)
        Note: for some reason, passing in the array as (n,8) results in a singularity, so we transpose twice
        """
      
        return dyn.calc_state_derivs(x.T, u.T, self.m, self.stiffness, self.damping, self.alpha, self.h, self.r).T
    
    def generate_random_state(self, initial_conditions:bool=False, n:int=1) -> np.ndarray:
        """
        Generate random state array
        :param initial_conditions: bool, if True, generate initial conditions, else generate random states
        :param n: int, number of states to generate
        :return: np.ndarray of size (n, num_states)
        """
        if type(self.xMin) == torch.Tensor:
            on_gpu = True
            self.xMin = self.xMin.cpu().numpy()
            self.xMax = self.xMax.cpu().numpy()
        else:
            on_gpu = False

        state = super().generate_random_state(n=n)

        if initial_conditions:
            state[:,:4] = np.ones((4,)) * 200
            state[:,4:6] = np.zeros((2,))

        if on_gpu:
            self.xMin = torch.from_numpy(self.xMin).cuda()
            self.xMax = torch.from_numpy(self.xMax).cuda()

        return state
    
    def generate_random_xgoal(self, n:int=1) -> np.ndarray:
        """
        Generate random state goal array
        :param n: int, number of states to generate
        :return: np.ndarray of size (n, num_states)
        """
        return self.generate_random_state(initial_conditions=True, n=n)

    def visualize(self, x:np.ndarray) -> None:
        """
        Visualize the grub
        :param x: state array, np.ndarray of size (num_data_points, 8)
        :param ax: matplotlib axis
        """
        if self.plotting:
            self.ax.clear()
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
            self.plotting = True

        q = x[0, 6:8]
        num_segments = 20
        last_pos = np.zeros([3, 1])
        for i in range(0, num_segments):
            scalar = float(i + 1) / num_segments
            this_pos = np.asarray(
                dyn.fkEnd(q * scalar, self.h * scalar, self.h * scalar)
            ).reshape(4, 4)[0:3, 3]
            last_pos = last_pos.flatten()
            self.ax.plot(
                    [last_pos[0], this_pos[0]],
                    [last_pos[1], this_pos[1]],
                    [last_pos[2], this_pos[2]],
                    color="orange",
                    linewidth=self.r * 100,
                    alpha=1.0,
                    )
            last_pos = this_pos

        self.ax.set_xlim3d([-self.h, self.h])
        self.ax.set_ylim3d([-self.h, self.h])
        self.ax.set_zlim3d([0, self.h])    
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")    
        plt.ion()
        plt.show()
        plt.pause(1e-6)
        return self.fig

    def visualize_horizon(self, x_history:np.ndarray, xgoal:np.ndarray, path:np.ndarray, horizon:int, sim_length:float, i:int) -> None:
        """
        Visualize the horizon
                :param x_history: state history array, np.ndarray of size (num_data_points, num_states)
        :param xgoal: state goal array, np.ndarray of size (num_data_points, num_states)
        :param path: planned state array, np.ndarray of size (num_data_points, num_states)
        :param horizon: int, horizon length
        :param sim_length: float, simulation length
        :param i: int, current time step
        :return: None
        """
        index = list(range(0, sim_length + horizon))

        plt.figure(f"{self.name} Horizon")
        plt.clf()
        plt.axhline(y=xgoal[6], color="y", linestyle="-")
        plt.axhline(y=xgoal[7], color="k", linestyle="-")
        plt.plot(index[i : i + horizon], path[:, 6], ":r", index[i : i + horizon], path[:, 7], ":b")
        plt.plot(index[:i], x_history[:i, 6], "-m", index[:i], x_history[:i, 7], "-c")
        plt.legend(["Goal U", "Goal V", "Planned U", "Planned V", "Sim U", "Sim V"])
        plt.xlabel("Time Step")
        plt.ylabel("Joint Angle (rad)")
        plt.grid(True)
        plt.show()
        plt.pause(1e-7)

if __name__ == "__main__":
    # example of how to use this file
    """
    This is a demonstration of how to use the GrubSim class and its methods
    """
    grub_params = {
    "mass": 5.3946339367881455,
    "stiffness": 12.657823577745912,
    "damping": 3.714360045199257,
    "pressure_resp_coeff": 3.386059758250698,
    "h": 0.1667074298414083,
    "r": 0.0688555835455514,
    }
    sys = GrubSim(**grub_params)
    x = sys.generate_random_state()
    u = sys.generate_random_command()

    dt = 0.01
    sim_time = 2
    horizon = int(sim_time / dt)

    x_history = np.zeros((horizon, sys.numStates))
    u_history = np.zeros((horizon, sys.numInputs))

    plt.ion()
    sys.visualize(x)

    for i in range(0, horizon):
        x = sys.forward_simulate_dt(x, u, dt)

        x_history[i, :] = x.flatten()
        u_history[i, :] = u.flatten()

        if i % 5 == 0:
            sys.visualize(x)

    print(x_history[-1, :])
    print(u_history[-1, :])
    sys.plot_history(x_history, u_history, np.zeros((1, 8)), block=True)
