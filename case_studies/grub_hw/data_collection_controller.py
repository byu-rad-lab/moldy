#!/usr/bin/env python3
#from .utils import randomizer
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend (non-interactive)
import matplotlib.pyplot as plt
import rospy
import numpy as np
import time
from typing import List
import sys
from moldy.validation.utils import start_bag_recording, stop_all_recording, send_steady_state_pressure_command

from bellows_arm.src.bellows_arm_hardware.bellows_hw_interface import BellowsHWInterface




class DataCollectionController(BellowsHWInterface):
    def __init__(self, joint_numbers: List[int], sensor_feedback: List[str]):
        self.min = 0  # kPa
        self.max = 275  # kPa
        super().__init__(joint_numbers, sensor_feedback, (self.min + self.max) / 2, namespace="/robo_0")
        self.num_joints = len(joint_numbers)
        self.commands = [np.zeros((4, 1)) for i in range(self.num_joints)]

    def get_random_commands(self):
        for i in range(self.num_joints):
            self.commands[i] = np.random.randint(self.min, high=self.max, size=(4,))
        return self.commands

def main_loop(command_path:str, record_path:str=None) -> None:
    """
    command_path: str - file name/path of a .npy array containing the pressure commands
    record_path: str - file name/path of the .bag file to record the data
    """
    
    print('Entering main loop')
    commands = np.load(command_path)

    joints = [0]

    rospy.init_node("data_collection_controller")
    print('ROS node initialized')
    sensor_feedback = ["pressure", "vive"]
    controller = DataCollectionController(joints, sensor_feedback)

    rate = rospy.Rate(100)
    i = 0

    if record_path is not None:
        start_bag_recording(record_path)

    send_steady_state_pressure_command(controller, rate)

    print("Starting ROS Loop")

    while not rospy.is_shutdown() and i < commands.shape[0]:
        u = commands[i]
        controller.send_pressure_commands([u])

        if i % 1000 == 0:
            print(f"Running Trajectory Step {i}/{commands.shape[0]}")

        i += 1
        rate.sleep()

    print("Finished Running Trajectory")

    send_steady_state_pressure_command(controller, rate)

    if record_path is not None:
        stop_all_recording()


if __name__ == "__main__":
    command_path = '/home/daniel/Documents/data/xfer_learning/grub_data_collection/test_dataset_commands.npy'
    record_path = '/home/daniel/Documents/data/xfer_learning/grub_data_collection/sideways_weight_grub_data_MAIN/asfasfdlasdfkjasdf.bag'
    main_loop(command_path, record_path)