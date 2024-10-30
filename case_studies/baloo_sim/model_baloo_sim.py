import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

from moldy.model.Model import Model

class BalooSim(Model):
    def __init__(
        self,
        # XML_PATH="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_baloo.xml",
        XML_PATH="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/baloo.xml",
        **kwargs,
    ):
        """
        Model of Baloo robot from BYU RaD Lab. This class will interface to the left arm only for now.
            3 Link Bellows Arm Simulation Model
                - State is 12 presssures, 6 positions, 6 velocities
                - Input is 12 pressure commands
            :param XML_PATH: str, path to mujoco xml file
        """
        self.numSims = kwargs.get("numSims", 1)
        kwargs.pop("numSims", None)

        super().__init__(**kwargs)

        self.name = "baloo_arm_sim"
        self.numStates = 8 * 3 # 12 pressures, 6 velocities, 6 positions
        self.numInputs = 4 * 3

        self.setup_mujoco(XML_PATH)

        self.mujoco_pressure_multiplier = 1
        self.mujoco_dt = 0.005
        self.max_pressure = 200 # params["general"]["max_pressure"] / self.mujoco_pressure_multiplier

        self.uMax = np.ones((1,self.numInputs))*self.max_pressure
        self.uMin = self.uMax*0.0

        self.xMax = np.array(
            [[self.max_pressure,self.max_pressure,self.max_pressure,self.max_pressure,
                self.max_pressure,self.max_pressure,self.max_pressure,self.max_pressure,
                self.max_pressure,self.max_pressure,self.max_pressure,self.max_pressure,    
                np.pi,np.pi,
                np.pi,np.pi,
                np.pi,np.pi,
                np.pi/2,np.pi/2,
                np.pi/2,np.pi/2,
                np.pi/2,np.pi/2
                ]])
        self.xMin = np.array(
            [[0,0,0,0,
              0,0,0,0,
              0,0,0,0,
              -np.pi,-np.pi,
              -np.pi,-np.pi,
              -np.pi,-np.pi,
              -np.pi/2,-np.pi/2,
              -np.pi/2,-np.pi/2,
              -np.pi/2,-np.pi/2,
              ]])


    def setup_mujoco(self, XML_PATH:str) -> None:
        """
        Setup the mujoco simulation
        :param XML_PATH: str, path to mujoco xml file
        """
        self.mujoco_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = [mujoco.MjData(self.mujoco_model) for _ in range(self.numSims)]
        self.num_disks = int(self.mujoco_model.numeric("num_disks").data.item())

    def set_joint_pressures(self, pressures:np.ndarray, side:str="left", i:int=0) -> None:
        """
        Set the pressures of the bellows joints
        :param pressures: np.ndarray, pressures to set for each joint
        :param side: str, which arm to set the pressures for (left or right)
        :param i: int, index of the simulation to set the pressures for
        
        pressures = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
        """
        # self.data[i].act = pressures * self.mujoco_pressure_multiplier

        for jointnum in range(3):
            for j in range(4):
                self.data[i].act[self.mujoco_model.actuator(f"{side}_{jointnum}_p{j}").actadr] = pressures[jointnum*4 + j] * self.mujoco_pressure_multiplier

    def get_joint_pressures(self, side:str="left", i:int=0) -> np.ndarray:
        """
        Get the pressures of the bellows joints
        :param side: str, which arm to get the pressures for (left or right)
        :param i: int, index of the simulation to get the pressures for
        :return: np.ndarray, pressures of the bellows joints (should be 1x12)
        """
        pressures = []
        for jointnum in range(3):
            for j in range(4):
                pressures.append(self.data[i].act[self.mujoco_model.actuator(f"{side}_{jointnum}_p{j}").actadr])
        return (np.asarray(pressures) / self.mujoco_pressure_multiplier).reshape(1, -1)

    def set_joint_angles(self, jangles:np.ndarray, side:str="left", i:int=0) -> None:
        """
        Set the joint angles of the arm
        :param jangles: np.ndarray, joint angles to set
        :param side: str, which arm to set the joint angles for (left or right)
        :param i: int, index of the simulation to set the joint angles for
        """
        for jointnum in range(3):
            x_disk_angle = jangles[jointnum*2] / (self.num_disks - 1)
            y_disk_angle = jangles[jointnum*2 + 1] / (self.num_disks - 1)
            for j in range(self.num_disks - 1):
                self.data[i].qpos[self.mujoco_model.joint(f"{side}_{jointnum}_Jx_{j}").id] = x_disk_angle
                self.data[i].qpos[self.mujoco_model.joint(f"{side}_{jointnum}_Jy_{j}").id] = y_disk_angle
                
    def get_joint_angles(self, side:str="left", i:int=0) -> np.ndarray:
        """
        Get the joint angles of the arm
        :param side: str, which arm to get the joint angles for (left or right)
        :param i: int, index of the simulation to get the joint angles for
        :return: np.ndarray, joint angles of the arm
        """
        jangles = []
        for jointnum in range(3):
            jangles.append(self.data[i].sensor(f"{side}_{jointnum}").data[0])
            jangles.append(self.data[i].sensor(f"{side}_{jointnum}").data[1])

        return np.asarray(jangles).reshape(1, -1)

    def set_joint_velocities(self, vel:np.ndarray, side:str="left", i:int=0) -> None:
        """
        Set the joint velocities of the arm
        :param vel: np.ndarray, joint velocities to set
        :param side: str, which arm to set the joint velocities for (left or right)
        :param i: int, index of the simulation to set the joint velocities for
        """
        for jointnum in range(3):
            x_disk_angle = vel[jointnum*2] / (self.num_disks - 1)
            y_disk_angle = vel[jointnum*2 + 1] / (self.num_disks - 1)
            for j in range(self.num_disks - 1):
                self.data[i].qvel[self.mujoco_model.joint(f"{side}_{jointnum}_Jx_{j}").dofadr] = x_disk_angle
                self.data[i].qvel[self.mujoco_model.joint(f"{side}_{jointnum}_Jy_{j}").dofadr] = y_disk_angle

    def get_joint_velocities(self, side:str="left", i:int=0) -> np.ndarray:
        """
        Get the joint velocities of the arm
        :param side: str, which arm to get the joint velocities for (left or right)
        :param i: int, index of the simulation to get the joint velocities for
        :return: np.ndarray, joint velocities of the arm
        """
        jvels = []
        for jointnum in range(3):
            jvels.append(self.data[i].sensor(f"{side}_{jointnum}").data[2])
            jvels.append(self.data[i].sensor(f"{side}_{jointnum}").data[3])

        return np.asarray(jvels).reshape(1, -1)

    def set_state(self, x:np.ndarray, i:int) -> None:
        """
        Set the state of the simulation
        :param x: np.ndarray, state to set
        :param i: int, index of the simulation to set the state for
        """
        if i == -1:
            for j in range(self.numSims):
                self.set_state(x, j)
            return
        self.set_joint_pressures(x[0, :12], i=i)
        self.set_joint_velocities(x[0, 12:18], i=i)
        self.set_joint_angles(x[0, -6:], i=i)
        # mj_forward(self.mujoco_model, self.data[i])

    def get_state(self, i:int=0) -> np.ndarray:
        """
        Get the state of the simulation
        :param i: int, index of the simulation to get the state for
        :return: np.ndarray, state of the simulation
        """
        pressures = self.get_joint_pressures(i=i)
        joint_vel = self.get_joint_velocities(i=i)
        joint_pos = self.get_joint_angles(i=i)
        return np.hstack([pressures, joint_vel, joint_pos])

    def set_pcmd(self, u:np.ndarray, side:str="left", i:int=0) -> None:
        """
        Set the pressure commands for the bellows joints
        :param u: np.ndarray, pressure commands to set
        :param side: str, which arm to set the pressure commands for (left or right)
        :param i: int, index of the simulation to set the pressure commands for
        """
        for jointnum in range(3):
            for j in range(4):
                self.data[i].ctrl[self.mujoco_model.actuator(f"{side}_{jointnum}_p{j}").id] = u[0, jointnum*4 + j] * self.mujoco_pressure_multiplier

    def forward_simulate_dt(self, x:np.ndarray, u:np.ndarray, dt:float, base_data:np.ndarray=None, iter:int=0) -> np.ndarray:
        """
        Forward simulate the simulation by dt. Doesn't use x because we assume that x is stored in data. Was having problems setting x directly without letting the robot settle.
        :param x: np.ndarray, state of the simulation
        :param u: np.ndarray, pressure commands
        :param dt: float, time to simulate forward by
        :param base_data: np.ndarray, data to use as the base for the simulation
        :param iter: int, iteration number
        :return: np.ndarray, next state of the simulation
        """
        u = np.clip(u, self.uMin, self.uMax) 

        next_states = np.zeros((self.numSims, self.numStates))

        if base_data is not None and iter == 0:
            for i in range(self.numSims):
                self.data[i] = deepcopy(base_data[0])

        for i in range(self.numSims):
            # self.set_state(x[i].reshape(1,self.numStates))
            self.set_pcmd(u[i].reshape(1,self.numInputs), i=i)

            for j in range(int(dt / self.mujoco_dt)):
                mujoco.mj_step(self.mujoco_model, self.data[i])
            next_states[i] = self.get_state(i)

        return next_states
    
    def forward_sim_user_input(self, x:np.ndarray, u:np.ndarray, dt:float) -> np.ndarray:
        """
        function that allows for user input into the mujoco simulation to allow plotting and interaction with the arm. Doesn't use x or u. u comes from the mujoco GUI.
        :param x: np.ndarray, state of the simulation
        :param u: np.ndarray, pressure commands
        :param dt: float, time to simulate forward by
        :return: np.ndarray, next state of the simulation
        """

        for i in range(int(dt / self.mujoco_dt)):
            mujoco.mj_step(self.mujoco_model, self.data[0])

        return self.get_state(0).reshape(1, self.numStates)

    def convert_diff_to_command(self, u):
        """
        u is shape (1,6) where there are two differential pressures for each joint
            u has values only between -1 and 1. 1 corresponds to p0=uMax and p1=uMin.
        Need to convert to (1,12) where each value is between uMin and uMax
        """     
        if len(u.shape) < 2:
            u = u.reshape(1, -1)

        if type(u) == torch.Tensor:
            u = u.cpu().detach().numpy()
        J = np.array([u[:, 0], -u[:, 0], u[:, 1], -u[:, 1], u[:, 2], -u[:, 2], u[:, 3], -u[:, 3], u[:, 4], -u[:, 4], u[:, 5], -u[:, 5]]).T
        if type(self.uMax) == torch.Tensor:
            J = torch.tensor(J).float().cuda()

        u = 0.5 * (J + 1) * (self.uMax - self.uMin) + self.uMin
        return u

    def forward_simulate_dt_differential(self, x, u, dt):
        """
        Forward simulate using differential pressures
        u is shape (1,6) where there are two differential pressures for each joint
        """
        u = deepcopy(u)

        u = self.convert_diff_to_command(u)

        next_states = self.forward_simulate_dt(x, u, dt)
        return next_states

    def generate_random_state(self, initial_conditions:bool=False, rand_pos:bool=True, n:int=1) -> np.ndarray:
        """
        Currently this function assumes
            - max/min pressures are the same for each bellow
            - max/min for u and v within each joint are the same
            - max/min for jts 1 and 2 are the same
        """
        if type(self.xMin) == torch.Tensor:
            on_gpu = True
            self.xMin = self.xMin.cpu()
            self.xMax = self.xMax.cpu()
        else:
            on_gpu = False

        state = super().generate_random_state(n=n)

        if initial_conditions:
            state[:, :12] = np.ones((n, 12)) * self.xMax[:,0]/2
            state[:, 12:18] = np.zeros((n, 6))

        if not rand_pos:
            state[:, -6:] = np.zeros((6, n))

        if on_gpu:
            self.xMin = self.xMin.cuda()
            self.xMax = self.xMax.cuda()
            state = state.cuda()

        return state

    def generate_random_xgoal(self, n:int=1) -> np.ndarray:
        """
        Generate a random xgoal using generate_random_state
        """
        self.xMax[0, 18:20] = 1.0
        self.xMin[0, 18:20] = -1.0

        self.xMax[0, -4:] = 1.25
        self.xMin[0, -4:] = -1.25

        xgoal = self.generate_random_state(initial_conditions=True, n=n)

        self.xMax[0, -6:] = np.pi/2
        self.xMin[0, -6:] = -np.pi/2

        return xgoal

    def visualize_horizon(self, x_history:np.ndarray, xgoal:np.ndarray, path:np.ndarray, horizon:int, sim_length:int, i:int) -> None:
        """
        Visualize the horizon of the simulation
        :param x_history: np.ndarray, history of the states
        :param xgoal: np.ndarray, goal state
        :param path: np.ndarray, path of the states
        :param horizon: int, length of the horizon
        :param sim_length: int, length of the simulation
        :param i: int, iteration number
        """
        index = list(range(0, sim_length + horizon))
        i += 1

        plt.figure(f"{self.name} Horizon - Joint 0")
        plt.clf()
        plt.axhline(y=xgoal[18], color="g", linestyle="-")
        plt.axhline(y=xgoal[19], color="y", linestyle="-")
        plt.plot(index[i-1 : i - 1 + horizon], path[:, 18], ":c", index[i-1 : i - 1 + horizon], path[:, 19], ":r")
        plt.plot(index[:i], x_history[:i, 18], "-c", index[:i], x_history[:i, 19], "-r")
        plt.legend(["Goal U", "Goal V", "Planned U", "Planned V", "Sim U", "Sim V"])
        plt.grid(True)

        plt.figure(f"{self.name} Horizon - Joint 1")
        plt.clf()
        plt.axhline(y=xgoal[20], color="g", linestyle="-")
        plt.axhline(y=xgoal[21], color="y", linestyle="-")
        plt.plot(index[i-1 : i - 1 + horizon], path[:, 20], ":c", index[i-1 : i - 1 + horizon], path[:, 21], ":r")
        plt.plot(index[:i], x_history[:i, 20], "-c", index[:i], x_history[:i, 21], "-r")
        plt.legend(["Goal U", "Goal V", "Planned U", "Planned V", "Sim U", "Sim V"])
        plt.grid(True)

        plt.figure(f"{self.name} Horizon - Joint 2")
        plt.clf()
        plt.axhline(y=xgoal[22], color="g", linestyle="-")
        plt.axhline(y=xgoal[23], color="y", linestyle="-")
        plt.plot(index[i-1 : i - 1 + horizon], path[:, 22], ":c", index[i-1 : i - 1 + horizon], path[:, 23], ":r")
        plt.plot(index[:i], x_history[:i, 22], "-c", index[:i], x_history[:i, 23], "-r")
        plt.legend(["Goal U", "Goal V", "Planned U", "Planned V", "Sim U", "Sim V"])
        plt.grid(True)

        plt.show()
        plt.pause(0.0001)

    def visualize(self, viewer:mujoco.viewer) -> None:
        """
        Visualize the simulation
        :param viewer: mujoco.viewer.MjViewer, viewer to visualize the simulation
        """
        viewer.sync()


if __name__ == "__main__":
    sys = BalooSim()

    x = sys.generate_random_state(initial_conditions=True)
    u = sys.generate_random_command()
    dt = 0.005
    sim_time = 1000
    horizon = int(sim_time / dt)

    x_history = np.zeros((horizon, sys.numStates))
    u_history = np.zeros((horizon, sys.numInputs))

    # create subplots for the joint angles
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    plt.ion()
    ax[0].set_title("Joint 0")
    ax[0].set_ylabel("Position (rad)")
    ax[1].set_title("Joint 1")
    ax[1].set_ylabel("Position (rad)")
    ax[2].set_title("Joint 2")
    ax[2].set_ylabel("Position (rad)")
    ax[2].set_xlabel("Time (s)")

    np.set_printoptions(precision=4, suppress=True)
    sys.set_state(x, 0)

    with mujoco.viewer.launch_passive(sys.mujoco_model, sys.data[0]) as viewer:
        for i in range(0, horizon):
            x = sys.forward_sim_user_input(x, u, dt)
            # x = sys.forward_simulate_dt(x, u, dt)
            x_history[i, :] = x.flatten()
            u_history[i, :] = u.flatten()

            if i % 5 == 0:
                sys.visualize(viewer)

                # print joint angles
                print(sys.get_joint_angles())

                ax[0].plot(range(i), x_history[:i, 18], "-c", label="Sim U")
                ax[0].plot(range(i), x_history[:i, 19], "-k", label="Sim V")
                ax[1].plot(range(i), x_history[:i, 20], "-c", label="Sim U")
                ax[1].plot(range(i), x_history[:i, 21], "-k", label="Sim V")
                ax[2].plot(range(i), x_history[:i, 22], "-c", label="Sim U")
                ax[2].plot(range(i), x_history[:i, 23], "-k", label="Sim V")

                plt.pause(0.0001)

    print(x)
    sys.plot_history(x_history, u_history, np.zeros((1, sys.numStates)))
    pass
