#!/usr/bin/env python3
import rospy
import numpy as np
import torch
from typing import List
from std_msgs.msg import Header

from rad_msgs.msg import BellowsArmState
from bellows_arm.src.bellows_arm_hardware.bellows_hw_interface import BellowsHWInterface
from moldy.validation.control.nempc.NonlinearEMPC import NonlinearEMPC as NEMPC
from moldy.case_studies.grub_hw.learnedModel_grub_hw import LearnedModel_GrubHW
from moldy.validation.utils import send_steady_state_pressure_command, start_bag_recording, stop_all_recording


pressure_max = 275

class NEMPC_Grub_HW(BellowsHWInterface):
    def __init__(self, joint_numbers: List[int], sensors: List[str], trial_dir:str, Q:np.ndarray=None, R:np.ndarray=None) -> None:
        """
        Set up the NEMPC controller for the grub.
        :param joint_numbers: List[int]: The joint numbers to control.
        :param sensors: List[str]: The sensors to use for feedback.
        :param trial_dir: The directory containing the learned model. If None, the analytical model is used.
        :param Q: The Q matrix for the NEMPC controller. Diagonal matrix of size (numStates, numStates).
        :param R: The R matrix for the NEMPC controller. Diagonal matrix of size (numInputs, numInputs).
        """
        self.min = 0  # kPa
        self.max = pressure_max
        super().__init__(joint_numbers, sensors, (self.min + self.max) / 2, namespace="robo_0")

        if Q is None:
            Q = 1.0 * np.diag([0, 0, 0, 0, 0.1, 0.1, 50.0, 50.0])
        if R is None:
            R = 1e-2*np.diag([1.0, 1.0, 1.0, 1.0])

        Q = torch.from_numpy(Q).float().cuda()
        R = torch.from_numpy(R).float().cuda()

        system = LearnedModel_GrubHW(trial_dir=trial_dir)

        def CostFunc(x:np.ndarray, u:np.ndarray, xgoal:np.ndarray, ugoal:np.ndarray, prev_u:np.ndarray=None, final_timestep:bool=False):
            """
            Cost function for the NEMPC controller.
            :param x: np.ndarray: Current state, size (1, numStates)
            :param u: np.ndarray: Current command, size (1, numInputs)
            :param xgoal: np.ndarray: Goal state, size (1, numStates)
            :param ugoal: np.ndarray: Goal command, size (1, numInputs)
            :param prev_u: np.ndarray: Previous command, size (1, numInputs)
            :param final_timestep: bool: Whether or not this is the final timestep.
            :return: np.ndarray: Cost, size (1, 1)
            """

            Qx = torch.mm((x - xgoal) ** 2.0, Q)
            Rx = torch.mm((u-prev_u) ** 2.0, R)
            cost = torch.sum(Qx, axis=1) + torch.sum(Rx, axis=1)
            return cost

        self.controller = NEMPC(
                            system.forward_simulate_dt,
                            CostFunc,
                            numStates=system.numStates,
                            numInputs=system.numInputs,
                            umin=system.uMin.detach().cpu().numpy(),
                            umax=system.uMax.detach().cpu().numpy(),
                            horizon=30,
                            dt=0.02,
                            numKnotPoints=1,
                            useGPU=True,
                            selection_method="tournament",
                            tournament_size=5,
                            crossover_method="knot_point",
                            mutation_probability=0.1,
                            numSims=1000,
                            numParents=300,
                            numStrangers=100,
                        )
        self.numInputs = system.numInputs
        self.numStates = system.numStates
        self.x = np.zeros((1, self.numStates))
        self.a = 1.0
        self.b = 1.0 - self.a

    def calc_pressure_commands(self, xgoal, ulast):

        self.pressures[0][self.pressures[0] < 0] = 0.0

        self.x[0, :4] = self.pressures[0]
        self.x[0, 4:6] = self.joint_vel[0]
        self.x[0, -2:] = self.joint_angles[0]

        u_nempc, path = self.controller.solve_for_next_u(
            self.x,
            xgoal,
            ulast=ulast,
            ugoal=np.zeros((1,self.numInputs)),
            mutation_noise=0.99,
        )

        # u_cmd1 = u_nempc[0,:].reshape(1,4)

        u_cmd = self.b*ulast + self.a*u_nempc[0,:].reshape(1,4)
        # print(u_cmd)
        # print(u_cmd == u_cmd1)

        # print(u_cmd1.shape, u_cmd.shape)
        # print(u_cmd)
        # print(u_cmd.shape)

        # print(u_cmd1 - u_cmd)

        return u_cmd.squeeze(), self.x

def main_loop(trajectory_path:str, trial_dir:str=None, record_path:str=None) -> None:
    """
    Run the NEMPC controller for the grub hardware using ROS.
    :param trial_dir: The directory containing the learned model. If None, the analytical model is used.
    :return: None
    """

    rospy.init_node("grub_nempc_controller")
    joints = [0]
    sensor_feedback = ["pressure", "vive"]
    controller = NEMPC_Grub_HW(joints, sensor_feedback, trial_dir)

    desired_xgoal_publisher = rospy.Publisher("/robo_0/joint_0/joint_cmd", BellowsArmState, queue_size=1)
    msg = BellowsArmState()
    trajectory = np.load(trajectory_path)
    u = np.zeros((1,4))

    rate = rospy.Rate(25)
    i = 0
    iae_error = 0


    if record_path is not None:
        start_bag_recording(record_path)

    send_steady_state_pressure_command(controller, rate, time_seconds=1)

    while not rospy.is_shutdown() and i < trajectory.shape[0]:
        xgoal = trajectory[i]

        u, x = controller.calc_pressure_commands(xgoal, u)

        controller.send_pressure_commands([u.T])

        h = Header()
        h.stamp = rospy.Time.now()
        msg.header = h
        msg.position = xgoal[-2:]
        desired_xgoal_publisher.publish(msg)

        iae_error += np.sum(np.abs(xgoal[-2:] - controller.joint_angles[0]))
        i += 1
        rate.sleep()

    print(f"Finished Running Trajectory. IAE Error: {iae_error}")
    send_steady_state_pressure_command(controller, rate, time_seconds=1)
    if record_path is not None:
        stop_all_recording()

if __name__ == "__main__":
    main_loop(
        trajectory_path="/home/daniel/catkin_ws/src/moldy/paper_figs_data/hw_control_commands/step_trajectory.npy",
        trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/results/best_models/version_17",
        record_path=None
    )
