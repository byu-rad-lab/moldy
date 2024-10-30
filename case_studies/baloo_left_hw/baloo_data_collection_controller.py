#!/usr/bin/env python3

import rospy
import numpy as np
from typing import List
import os
import time

from moldy.validation.utils import start_bag_recording, stop_all_recording, send_steady_state_pressure_command
from bellows_arm.src.bellows_arm_hardware.bellows_hw_interface import BellowsHWInterface

class DataCollectionController(BellowsHWInterface):
    def __init__(self, joint_numbers: List[int], sensor_feedback: List[str]):
        self.min = 0  # kPa
        self.max = 200  # kPa
        super().__init__(joint_numbers, sensor_feedback, (self.min + self.max) / 2, namespace="left_arm")
        self.num_joints = len(joint_numbers)
        self.commands = [np.zeros((4, 1)) for i in range(self.num_joints)]    

def create_trajectory(start_pressure_commands: np.ndarray, end_pressure_commands: np.ndarray, time_for_commands_seconds: float, knotpoints: int) -> np.ndarray:
    """
    start_pressure_commands: np.ndarray - 2D array of pressure commands to start the trajectory, (1, 12)
    end_pressure_commands: np.ndarray - 2D array of pressure commands to end the trajectory, (1, 12)
    time_for_commands_seconds: float - time in seconds to transition from start to end pressure commands
    """
    num_points = int(time_for_commands_seconds * 50) # TODO WE ARE HARDCODING 50 Hz RIGHT NOW
    trajectory = np.zeros((num_points, start_pressure_commands.shape[0], start_pressure_commands.shape[1]))

    for i in range(start_pressure_commands.shape[1]):
        knot_length = num_points // knotpoints
        knot_values = np.linspace(start_pressure_commands[:, i], end_pressure_commands[:, i], knotpoints)
        for j in range(knotpoints):
            trajectory[j*knot_length:(1+j)*knot_length, :, i] = knot_values[j]
    return trajectory

def perturbate_trajectory(start_command, end_command):
    perturbation = np.random.randint(-10, high=10, size=(2, 3, 4))

    start_command += perturbation[0]
    end_command += perturbation[1]

    start_command = np.clip(start_command, 0, 200)
    end_command = np.clip(end_command, 0, 200)

    return start_command, end_command

def main_loop(pressure_trajectory:np.ndarray, record_path:str=None, zero_cmd_time:float=10.0) -> None:
    """
    command_path: str - file name/path of a .npy array containing the pressure commands
    record_path: str - file name/path of the .bag file to record the data
    """
    
    print('Entering main loop')
    print('command array has shape: ', pressure_trajectory.shape)

    joints = [0, 1, 2]

    rospy.init_node("data_collection_controller")
    print('ROS node initialized')
    sensor_feedback = ["pressure", "vive"]
    controller = DataCollectionController(joints, sensor_feedback)

    rate = rospy.Rate(50)
    i = 0

    if record_path is not None:
        start_bag_recording(record_path)

    u0 = pressure_trajectory[0, 0]
    u1 = pressure_trajectory[0, 1]
    u2 = pressure_trajectory[0, 2]
    send_steady_state_pressure_command(controller, rate, command=[u0, u1, u2], time_seconds=10.0)

    print("Starting ROS Loop")

    while not rospy.is_shutdown() and i < pressure_trajectory.shape[0]:
        u0 = pressure_trajectory[i, 0]
        u1 = pressure_trajectory[i, 1]
        u2 = pressure_trajectory[i, 2]
        controller.send_pressure_commands([u0, u1, u2])

        i += 1
        rate.sleep()

    send_steady_state_pressure_command(controller, rate, command=[u0, u1, u2], time_seconds=10.0)
    send_steady_state_pressure_command(controller, rate, command=[np.zeros((4,1))]*3, time_seconds=zero_cmd_time)

    print("Finished Running Trajectory")
    if record_path is not None:
        stop_all_recording()


def get_random_pressure_command() -> np.ndarray:
    valid_pressures = False
    max_pressures = 200
    while not valid_pressures:
        u0 = np.random.randint(0, max_pressures, size=(4,1))
        # u0[[0, 1]] = np.random.randint(140, max_pressures, size=(2,1))
        # u0[[2, 3]] = np.random.randint(0, 60, size=(2,1))
        u1 = np.random.randint(0, max_pressures, size=(4,1))
        u0[[0, 1]] = np.random.randint(125, max_pressures, size=(2,1))
        u0[[2, 3]] = np.random.randint(0, 100, size=(2,1))
        u2 = np.random.randint(0, max_pressures, size=(4,1))

        # check pressure differentials
        u0_diff = u0[[0,2]] - u0[[1,3]]
        u1_diff = u1[[0,2]] - u1[[1,3]]
        u2_diff = u2[[0,2]] - u2[[1,3]]

        # print(u0, u0_diff)
        # print(u1, u1_diff)
        # print(u2, u2_diff)
        if (not (u1_diff[0] < -150 and u1_diff[1] > 150)) or ((not (u2_diff[0] < -150 and u2_diff[1] > 150)) or (not (u0_diff[0] < -150 and u0_diff[1] > 150))):
            valid_pressures = True
            print('valid pressures')
        else:
            print('invalid pressures')
    return [u0, u1, u2]

def main_loop_random_step_commands(record_path:str=None) -> None:
    """
    command_path: str - file name/path of a .npy array containing the pressure commands
    record_path: str - file name/path of the .bag file to record the data
    """
    
    print('Entering main loop')

    rospy.init_node("data_collection_controller")
    print('ROS node initialized')

    joints = [0, 1, 2]
    sensor_feedback = ["pressure", "vive"]
    controller = DataCollectionController(joints, sensor_feedback)

    rate = rospy.Rate(100)

    if record_path is not None:
        start_bag_recording(record_path)

    send_steady_state_pressure_command(controller, rate, command=[np.zeros((4,1))]*3, time_seconds=10.0)

    print("Starting ROS Loop")
    command_list = get_random_pressure_command()

    CMD_TIME_S = 10.0
    curr_time = time.time()
    RUN_TIME = 1800.0 
    start = time.time()

    while not rospy.is_shutdown() and time.time() - start < RUN_TIME:
        if time.time() - curr_time > CMD_TIME_S:
            command_list = get_random_pressure_command()
            curr_time = time.time()
            CMD_TIME_S = np.random.randint(5, 20)
            print(f"Run Time: {time.time() - start} seconds")

        controller.send_pressure_commands(command_list)
        rate.sleep()

    print("Finished, setting to 0 pressures")
    send_steady_state_pressure_command(controller, rate, command=[np.zeros((4,1))]*3, time_seconds=20.0)
    print("Finished Running Trajectory")

    if record_path is not None:
        stop_all_recording()


if __name__ == "__main__":
    record_path = f'/home/daniel/Documents/data/daniel_baloo_data_collection/noisy_pressure_data/trial7_30min.bag'
    main_loop_random_step_commands(record_path)


    # need to collect 10 trials...