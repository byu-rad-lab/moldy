#!/usr/bin/env python3

import rospy
import numpy as np
import time
from datetime import datetime
from std_msgs.msg import Header
import os
from typing import Tuple

from moldy.case_studies.grub_hw.nempc_grub_hw import NEMPC_Grub_HW
from rad_msgs.msg import BellowsArmState
from moldy.validation.utils import send_steady_state_pressure_command, start_bag_recording, stop_all_recording

"""
This script sends pressure commands to the physical robot, similar to the ControlTest functionality for the 
simulation case studies. You can load custom trajectories as seen below. 
"""

def controller_setup(trial_dir:str) -> NEMPC_Grub_HW:
    """
    Set up the NEMPC controller for the grub.
    :param trial_dir: The directory containing the learned model. If None, the analytical model is used.
    :return: The NEMPC controller.

    Currently set up to use only one joint (Grub)
    """
    joints = [0]
    sensor_feedback = ["pressure", "vive"]
    return NEMPC_Grub_HW(joints, sensor_feedback, trial_dir)

def test_setup() -> Tuple[rospy.Rate, np.ndarray, BellowsArmState]:
    """
    Set up the test.
    :return: Tuple[rospy.Rate, np.ndarray, BellowsArmState]:
        rate: rospy.Rate: ROS rate
        xgoal: np.ndarray: Goal states, size (numStates, 1)
        msg: BellowsArmState: ROS message
    """
    rate = rospy.Rate(50)
    xgoal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 8)
    msg = BellowsArmState()
    return rate, xgoal, msg

def run_trajectory(controller:NEMPC_Grub_HW, desired_xgoal_publisher:rospy.Publisher, trajectory:np.ndarray) -> None:
    """
    Run a trajectory on the grub.
    :param controller: NEMPC_Grub_HW: The NEMPC controller.
    :param desired_xgoal_publisher: rospy.Publisher: ROS publisher
    :param trajectory: np.ndarray: Trajectory to run, size (n, 8)
    :return: None
    """
    rate, xgoal, msg = test_setup()

    send_steady_state_pressure_command(controller, rate, time_seconds=15)
    i = 0
    j = 0
    u = np.zeros((1,4))
    iae = np.zeros((2,))

    print("Starting trajectory")
    while i < trajectory.shape[0] and not rospy.is_shutdown():
        start = time.time()
        xgoal[0, -2:] = trajectory[i, -2:]
        u, x = controller.calc_pressure_commands(trajectory[i].reshape(1, -1), u.reshape((1,4)))
        controller.send_pressure_commands([u])

        h = Header()
        h.stamp = rospy.Time.now()
        msg.header = h
        msg.position = xgoal[0, -2:]
        desired_xgoal_publisher.publish(msg)

        # print(f"Time to calculate and send u: {time.time() - start} s")
        if (time.time() - start) > 0.025:
            j += 1
        i += 1

        iae += np.abs(xgoal[0, -2:] - x[0, -2:])
        rate.sleep()

    print(f"Finished Running Trajectory. Number of slow calculations: {j}. IAE Error: {iae}")

def main_loop(trajectory:np.ndarray, logs_path:str, save_path:str) -> None:
    """
    The main loop of the test.
    :param logs_path: str: The path to the logs.
    :param save_path: str: The path to save the results.
    :return: None
    """
    rospy.init_node("grub_nempc_controller")
    desired_xgoal_publisher = rospy.Publisher("/robo_0/joint_0/joint_cmd", BellowsArmState, queue_size=1)

    test_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{save_path}_{test_name}"
    os.mkdir(save_path)

    for run in os.listdir(logs_path):
        controller = controller_setup(logs_path+"/"+run+"/")
    
        bag_filename = f"{save_path}/{run}_step.bag"
        start_bag_recording(bag_filename)
        run_trajectory(controller, desired_xgoal_publisher, trajectory)
        stop_all_recording()


if __name__ == "__main__":
    trajectory = np.load("/home/daniel/Documents/data/xfer_learning/grub_data_collection/xfer_paper_joint_commands.npy")

    # trajectory = np.zeros((500, 8))
    # trajectory[:, -2] = 0.5
    # run_name = "50K BASE"
    # run_name = "50K FT"
    # run_name = "100K BASE"
    # run_name = "100K FT"
    # run_name = "200K BASE"
    # run_name = "200K FT"
    # run_name = "360K BASE"
    # run_name = "360K FT"
    run_name = "Source"

    logs_path = f"/home/daniel/Documents/data/xfer_learning/sim_to_hw_grub_data/best_models/{run_name}"
    trial_name = f"sim_to_hw_{run_name}"
    record_path = f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/results/test_results/HW_CONTROL_TRIAL_{trial_name}"

    main_loop(trajectory, logs_path, record_path)