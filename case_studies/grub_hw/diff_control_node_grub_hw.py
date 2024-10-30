#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Header

from rad_msgs.msg import BellowsArmState
from bellows_arm.src.bellows_hw_interface import BellowsHWInterface
from moldy.validation.control.differential_pid.differential_controller import DifferentialController
from validation.utils import start_bag_recording, stop_all_recording, send_steady_state_pressure_command

def main_loop(min_pressure:float, max_pressure:float, 
              kp:np.ndarray, kd:np.ndarray, ki:np.ndarray, 
              rateHz:int, traj_path:str,
              record_path:str) -> None:
    rospy.init_node("grub_differential_controller")

    hw_interface = BellowsHWInterface([0], ["pressure", "vive"], (min_pressure+max_pressure)/2)
    controller = DifferentialController(kp, kd, ki, min_pressure, max_pressure, Ts=1/rateHz)

    rate = rospy.Rate(rateHz)

    desired_xgoal_publisher = rospy.Publisher("/robo_0/joint_0/joint_cmd", BellowsArmState, queue_size=1)
    msg = BellowsArmState()
    trajectory = np.load(traj_path)
    i = 0
    iae_error = 0

    if record_path is not None:
        start_bag_recording(record_path)
    
    send_steady_state_pressure_command(hw_interface, rate, time_seconds=2)

    while not rospy.is_shutdown() and i < trajectory.shape[0]:
        xgoal = trajectory[i]
        u_delta = controller.calc_delta_u(hw_interface.joint_angles[0], xgoal[-2:])
        u = controller.calc_final_u(u_delta).T

        hw_interface.send_pressure_commands([u])

        h = Header()
        h.stamp = rospy.Time.now()
        msg.header = h
        msg.position = xgoal[-2:]
        desired_xgoal_publisher.publish(msg)

        iae_error += np.sum(np.abs(xgoal[-2:] - hw_interface.joint_angles[0]))

        i += 1
        rate.sleep()

    print(f"Finished Running Trajectory. IAE Error: {iae_error}")
    send_steady_state_pressure_command(hw_interface, rate, time_seconds=2)

    if record_path is not None:
        stop_all_recording()


if __name__ == "__main__":
    min = 0  # kPa
    max = 275
    rate = 25

    kp = np.diag([95.0, 95.0])
    kd = -np.diag([0.3, 0.3])
    ki = np.diag([0.5, 0.5])*0.0

    trajectory_path = "/home/daniel/catkin_ws/src/moldy/paper_figs_data/hw_control_commands/step_trajectory.npy"
    record_path = "/home/daniel/catkin_ws/src/moldy/DATA_TEMP/data_diff_control_paper.bag"
    main_loop(min, max, kp, kd, ki, rate, trajectory_path, record_path=record_path)