import rospkg
import rospy
import subprocess
import os
import time
import numpy as np
from typing import List, Tuple

from bellows_arm.src.bellows_arm_hardware.bellows_hw_interface import BellowsHWInterface

def send_steady_state_pressure_command(interface:BellowsHWInterface, rate:rospy.Rate, command:np.ndarray=np.zeros((1,4)), time_seconds:float=15) -> None:
    """
    Send a steady state command to the grub.
    :param controller: NEMPC_Grub_HW: The NEMPC controller.
    :param rate: rospy.Rate: ROS rate
    :param time_seconds: float: The amount of time to send the pressure command for.
    :return: None
    """
    start = time.time()
    while (time.time() - start) < time_seconds and not rospy.is_shutdown():
        interface.send_pressure_commands(command)
        rate.sleep()

def start_bag_recording(bag_filename:str) -> None:
    """
    Start recording a rosbag.
    :param bag_filename: str: The filename of the rosbag.
    :return: None"""
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('moldy')
    cmd = [
        'rosbag',
        'record',
        '-O', bag_filename,
        '-a',
        '-q', 
    ]
    subprocess.Popen(cmd, cwd=package_path)

def stop_all_recording() -> None:
    """
    Stop recording ALL rosbags.
    """
    node_list = os.popen('rosnode list').read().splitlines()
    record_nodes = [node for node in node_list if node.startswith('/record_')]
    for node in record_nodes:
        os.system(f'rosnode kill {node}')


def calculate_trial_iae(results:dict, states_of_interest:List[int])-> List[Tuple[str, float]]:
    """
    Get the IAEs of the trials
    :param results: dict: The results of the trials
    :param states_of_interest: List[int]: The states to get the IAEs of
    
    :return: List[Tuple[str, float]]: The IAEs of the trials
    """
    trials = list(results.keys())
    iaes = [np.sum(np.abs(results[key]["iae"][0, states_of_interest])) for key in trials if key != "ground_truth"]
    
    try:
        trials.remove("ground_truth")
    except:
        pass

    sorted_iaes = sorted(zip(trials, iaes), key=lambda x: x[1])
    print("Sorted IAEs")
    for trial, iae in sorted_iaes:
        print(f"Trial {trial}: {iae}")

    return sorted_iaes