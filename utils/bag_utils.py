from bagpy import bagreader
from typing import List
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

import os



def get_angle_data(b:bagreader, topic_names:List[str]):
    """
    Extracts angle information from bagreader for n number of joints
    :param: b bagreader object containing recorded data
    :topic_names:List[str] name of each topic for which to extract angle data
    
    :return timestamps for angle data, u angle positions, v angle positions, u angle velocity, v angle velocity
    """

    t_angles = []
    us = []
    vs = []
    u_dots = []
    v_dots = []

    for topic in topic_names:
        try:
            filename = b.datafolder + topic + '.csv'
            df_angle_state = pd.read_csv(filename)  
        except:
            topic_name = topic.replace('-', '/')
            angle_data = b.message_by_topic(topic_name)
            df_angle_state = pd.read_csv(angle_data)

        t_angles.append(df_angle_state['Time'].to_numpy())
        us.append(df_angle_state['position_0'].to_numpy())
        vs.append(df_angle_state['position_1'].to_numpy())
        u_dots.append(df_angle_state['velocity_0'].to_numpy())
        v_dots.append(df_angle_state['velocity_1'].to_numpy())

    return t_angles, us, vs, u_dots, v_dots

def get_joint_command(b:bagreader, topic_names:List[str]):
    """
    Extracts angle information from bagreader for n number of joints
    :param: b bagreader object containing recorded data
    :topic_names:List[str] name of each topic for which to extract angle data
    
    :return timestamps for command data, u commands, v commands
    """

    t_cmds = []
    u_cmds = []
    v_cmds = []

    for topic in topic_names:
        try:
            filename = b.datafolder + topic + '.csv'
            df_joint_cmd = pd.read_csv(filename)
        except:
            topic_name = topic.replace('-', '/')
            joint_cmd_data = b.message_by_topic(topic_name)
            df_joint_cmd = pd.read_csv(joint_cmd_data)

        t_cmds.append(df_joint_cmd['Time'].to_numpy())
        u_cmds.append(df_joint_cmd['position_0'].to_numpy())
        v_cmds.append(df_joint_cmd['position_1'].to_numpy())

    return t_cmds, u_cmds, v_cmds

def get_pressure_cmd(b:bagreader, topic_names:List[str]):
    """
    Extracts pressure command information from bagreader for n number of joints
    :param: b bagreader object containing recorded data
    :topic_names:List[str] name of each topic for which to extract angle data
    
    :return timestamps and commands for pressure command data, p0, p1, p2, p3
    """

    t_pressure_cmds = []
    p0_cmds = []
    p1_cmds = []
    p2_cmds = []
    p3_cmds = []

    for topic in topic_names:
        try:
            filename = b.datafolder + topic + '.csv'
            df_pressure_cmd = pd.read_csv(filename)
        except:
            topic_name = topic.replace('-', '/')
            pressure_cmd_data = b.message_by_topic(topic_name)
            df_pressure_cmd = pd.read_csv(pressure_cmd_data)

        t_pressure_cmds.append(df_pressure_cmd['Time'].to_numpy())
        p0_cmds.append(df_pressure_cmd['pressure_0'].to_numpy())
        p1_cmds.append(df_pressure_cmd['pressure_1'].to_numpy())
        p2_cmds.append(df_pressure_cmd['pressure_2'].to_numpy())
        p3_cmds.append(df_pressure_cmd['pressure_3'].to_numpy())

    return t_pressure_cmds, p0_cmds, p1_cmds, p2_cmds, p3_cmds

def get_pressure_data(b:bagreader, topic_names:List[str]):
    """
    Extracts pressure state information from bagreader for n number of joints
    :param: b bagreader object containing recorded data
    :topic_names:List[str] name of each topic for which to extract angle data
    
    :return timestamps and states for pressure state data, p0, p1, p2, p3
    """

    t_pressure_states = []
    p0_states = []
    p1_states = []
    p2_states = []
    p3_states = []

    for topic in topic_names:
        try:
            filename = b.datafolder + topic + '.csv'
            df_pressure_state = pd.read_csv(filename)
        except:
            topic_name = topic.replace('-', '/')
            pressure_state_data = b.message_by_topic(topic_name)
            df_pressure_state = pd.read_csv(pressure_state_data)

        t_pressure_states.append(df_pressure_state['Time'].to_numpy())
        p0_states.append(df_pressure_state['pressure_0'].to_numpy())
        p1_states.append(df_pressure_state['pressure_1'].to_numpy())
        p2_states.append(df_pressure_state['pressure_2'].to_numpy())
        p3_states.append(df_pressure_state['pressure_3'].to_numpy())

    return t_pressure_states, p0_states, p1_states, p2_states, p3_states

def dirty_derivative_pressures(pressure_t, p0, p1, p2, p3, plot=False):
    sigma = 0.001
    dataset_size = p0.shape[0]

    p0_dot = np.zeros((dataset_size))
    p1_dot = np.zeros((dataset_size))
    p2_dot = np.zeros((dataset_size))
    p3_dot = np.zeros((dataset_size))

    for i in range(1, dataset_size):
        Ts  = (pressure_t[i] - pressure_t[i-1])
        denom = (2*sigma + Ts)
        beta = ((2*sigma - Ts) / denom)
        p0_dot[i] = (beta * p0_dot[i-1]) + (2 / denom) * (p0[i] - p0[i-1])
        p1_dot[i] = (beta * p1_dot[i-1]) + (2 / denom) * (p1[i] - p1[i-1])
        p2_dot[i] = (beta * p2_dot[i-1]) + (2 / denom) * (p2[i] - p2[i-1])
        p3_dot[i] = (beta * p3_dot[i-1]) + (2 / denom) * (p3[i] - p3[i-1])

    return p0_dot, p1_dot, p2_dot, p3_dot

def trim_data(t:np.ndarray, data:np.ndarray, tspan:List[float]):
    """
    Trims data and associated timestamps to be in the range of tspan
    :param t: timestamps for the data
    :param data: data to be trimmed
    :param tspan: [low, high] range of timestamps to keep

    :return t_trimmed, data_trimmed
    """
    idx = np.where((t >= tspan[0]) & (t <= tspan[1]))
    return t[idx], data[idx]

def get_trimmed_data_joint_command(tspan:List[float], t_angle: List[np.ndarray], t_cmd: List[np.ndarray], 
                                   u: List[np.ndarray], v: List[np.ndarray], 
                                   u_dot: List[np.ndarray], v_dot: List[np.ndarray], 
                                   u_cmd: List[np.ndarray], v_cmd: List[np.ndarray]) -> tuple:
    """
    Trims data and associated timestamps for joint commands to the overlapping time span of angle measurements and commands.
    :param t_angle: List of numpy arrays representing timestamps for angle measurements.
    :param t_cmd: List of numpy arrays representing timestamps for commands.
    :param u: List of numpy arrays representing angle measurements for joint u.
    :param v: List of numpy arrays representing angle measurements for joint v.
    :param u_dot: List of numpy arrays representing angular velocities for joint u.
    :param v_dot: List of numpy arrays representing angular velocities for joint v.
    :param u_cmd: List of numpy arrays representing commands for joint u.
    :param v_cmd: List of numpy arrays representing commands for joint v.

    :return: Trimmed timestamps and data for angle positions, velocities and commands.
    """
    t_angle_trimmed_list = []
    u_trimmed_list = []
    v_trimmed_list = []
    u_dot_trimmed_list = []
    v_dot_trimmed_list = []
    u_cmd_trimmed_list = []
    v_cmd_trimmed_list = []
    t_cmd_trimmed_list = []

    for i in range(len(u)):
        _, u_trimmed = trim_data(t_angle[i], u[i], tspan)
        _, v_trimmed = trim_data(t_angle[i], v[i], tspan)
        t_angle_trimmed, u_dot_trimmed = trim_data(t_angle[i], u_dot[i], tspan)
        t_angle_trimmed, v_dot_trimmed = trim_data(t_angle[i], v_dot[i], tspan)
        _, u_cmd_trimmed = trim_data(t_cmd[i], u_cmd[i], tspan)
        t_cmd_trimmed, v_cmd_trimmed = trim_data(t_cmd[i], v_cmd[i], tspan)

        t_angle_trimmed_list.append(t_angle_trimmed)
        u_trimmed_list.append(u_trimmed)
        v_trimmed_list.append(v_trimmed)
        u_dot_trimmed_list.append(u_dot_trimmed)
        v_dot_trimmed_list.append(v_dot_trimmed)
        u_cmd_trimmed_list.append(u_cmd_trimmed)
        v_cmd_trimmed_list.append(v_cmd_trimmed)
        t_cmd_trimmed_list.append(t_cmd_trimmed)

    return t_angle_trimmed_list, u_trimmed_list, v_trimmed_list, u_dot_trimmed_list, \
            v_dot_trimmed_list, u_cmd_trimmed_list, v_cmd_trimmed_list, t_cmd_trimmed_list

def get_trimmed_data_collection(tspan:List[float], t_angle: List[np.ndarray],
                                   u: List[np.ndarray], v: List[np.ndarray], 
                                   u_dot: List[np.ndarray], v_dot: List[np.ndarray]) -> tuple:
    """
    Trims data and associated timestamps for joint commands to the overlapping time span of angle measurements and commands.
    :param t_angle: List of numpy arrays representing timestamps for angle measurements.
    :param u: List of numpy arrays representing angle measurements for joint u.
    :param v: List of numpy arrays representing angle measurements for joint v.
    :param u_dot: List of numpy arrays representing angular velocities for joint u.
    :param v_dot: List of numpy arrays representing angular velocities for joint v.

    :return: Trimmed timestamps and data for angle positions, velocities and commands.
    """
    t_angle_trimmed_list = []
    u_trimmed_list = []
    v_trimmed_list = []
    u_dot_trimmed_list = []
    v_dot_trimmed_list = []

    for i in range(len(u)):
        t_angle_trimmed, u_trimmed = trim_data(t_angle[i], u[i], tspan)
        _, v_trimmed = trim_data(t_angle[i], v[i], tspan)
        _, u_dot_trimmed = trim_data(t_angle[i], u_dot[i], tspan)
        _, v_dot_trimmed = trim_data(t_angle[i], v_dot[i], tspan)

        t_angle_trimmed_list.append(t_angle_trimmed)
        u_trimmed_list.append(u_trimmed)
        v_trimmed_list.append(v_trimmed)
        u_dot_trimmed_list.append(u_dot_trimmed)
        v_dot_trimmed_list.append(v_dot_trimmed)

    return t_angle_trimmed_list, u_trimmed_list, v_trimmed_list, u_dot_trimmed_list, v_dot_trimmed_list

def get_trimmed_data_pressure(tspan:List[float], t_pressure: List[np.ndarray], p0: List[np.ndarray], p1: List[np.ndarray], p2: List[np.ndarray], p3: List[np.ndarray],
                              t_pressure_cmd:List[np.ndarray], pcmd_0:List[np.ndarray], pcmd_1:List[np.ndarray], pcmd_2:List[np.ndarray], pcmd_3:List[np.ndarray]) -> tuple:
    """
    Trims data and associated timestamps for pressure measurements to the overlapping time span of pressure measurements and commands.
    :param t_pressure: List of numpy arrays representing timestamps for pressure measurements.
    :param p0: List of numpy arrays representing pressure measurements for joint 0.
    :param p1: List of numpy arrays representing pressure measurements for joint 1.
    :param p2: List of numpy arrays representing pressure measurements for joint 2.
    :param p3: List of numpy arrays representing pressure measurements for joint 3.
    :param t_pressure_cmd: List of numpy arrays representing timestamps for pressure commands.
    :param p_cmd0: List of numpy arrays representing pressure commands for joint 0.
    :param p_cmd1: List of numpy arrays representing pressure commands for joint 1.
    :param p_cmd2: List of numpy arrays representing pressure commands for joint 2.
    :param p_cmd3: List of numpy arrays representing pressure commands for joint 3.

    :return: Trimmed timestamps and data for pressure measurements and commands.
    """
    t_pressure_trimmed_list = []
    p0_trimmed_list = []
    p1_trimmed_list = []
    p2_trimmed_list = []
    p3_trimmed_list = []
    t_pressure_cmd_trimmed_list = []
    p_cmd0_trimmed_list = []
    p_cmd1_trimmed_list = []
    p_cmd2_trimmed_list = []
    p_cmd3_trimmed_list = []

    for i in range(len(p0)):
        t_pressure_trimmed, p0_trimmed = trim_data(t_pressure[i], p0[i], tspan)
        _, p1_trimmed = trim_data(t_pressure[i], p1[i], tspan)
        _, p2_trimmed = trim_data(t_pressure[i], p2[i], tspan)
        _, p3_trimmed = trim_data(t_pressure[i], p3[i], tspan)
        t_cmd_trimmed, p_cmd0_trimmed = trim_data(t_pressure_cmd[i], pcmd_0[i], tspan)
        _, p_cmd1_trimmed = trim_data(t_pressure_cmd[i], pcmd_1[i], tspan)
        _, p_cmd2_trimmed = trim_data(t_pressure_cmd[i], pcmd_2[i], tspan)
        _, p_cmd3_trimmed = trim_data(t_pressure_cmd[i], pcmd_3[i], tspan)

        t_pressure_trimmed_list.append(t_pressure_trimmed)
        p0_trimmed_list.append(p0_trimmed)
        p1_trimmed_list.append(p1_trimmed)
        p2_trimmed_list.append(p2_trimmed)
        p3_trimmed_list.append(p3_trimmed)
        t_pressure_cmd_trimmed_list.append(t_cmd_trimmed)
        p_cmd0_trimmed_list.append(p_cmd0_trimmed)
        p_cmd1_trimmed_list.append(p_cmd1_trimmed)
        p_cmd2_trimmed_list.append(p_cmd2_trimmed)
        p_cmd3_trimmed_list.append(p_cmd3_trimmed)
    
    return t_pressure_trimmed_list, p0_trimmed_list, p1_trimmed_list, p2_trimmed_list, p3_trimmed_list, \
            t_pressure_cmd_trimmed_list, p_cmd0_trimmed_list, p_cmd1_trimmed_list, p_cmd2_trimmed_list, p_cmd3_trimmed_list

def resample_data_for_same_length_joint(t_angle_trimmed_list: List[np.ndarray], t_cmd_trimmed_list: List[np.ndarray],
                                  u_trimmed_list: List[np.ndarray], v_trimmed_list: List[np.ndarray],
                                  u_dot_trimmed_list: List[np.ndarray], v_dot_trimmed_list: List[np.ndarray],
                                  u_cmd_trimmed_list: List[np.ndarray], v_cmd_trimmed_list: List[np.ndarray], trimmed_tspan:List[float],
                                  rate:float) -> tuple:
    """
    Resamples data to be the same length.
    :param t_angle_trimmed_list: List of numpy arrays representing trimmed timestamps for angle measurements.
    :param t_cmd_trimmed_list: List of numpy arrays representing trimmed timestamps for commands.
    :param u_trimmed_list: List of numpy arrays representing trimmed angle measurements for joint u.
    :param v_trimmed_list: List of numpy arrays representing trimmed angle measurements for joint v.
    :param u_dot_trimmed_list: List of numpy arrays representing trimmed angular velocities for joint u.
    :param v_dot_trimmed_list: List of numpy arrays representing trimmed angular velocities for joint v.
    :param u_cmd_trimmed_list: List of numpy arrays representing trimmed commands for joint u.
    :param v_cmd_trimmed_list: List of numpy arrays representing trimmed commands for joint v.
    :param trimmed_tspan: List of floats representing the trimmed time span.
    :return: Tuple of resampled timestamps and resampled data for angle measurements, velocities, and commands.
    """
    # Resample the timestamps to be the same length
    t_resampled = np.arange(trimmed_tspan[0], trimmed_tspan[1], rate)
    
    # Interpolate data to the resampled timestamps
    u_resampled_list = [PchipInterpolator(t_angle, u)(t_resampled) for t_angle, u in zip(t_angle_trimmed_list, u_trimmed_list)]
    v_resampled_list = [PchipInterpolator(t_angle, v)(t_resampled) for t_angle, v in zip(t_angle_trimmed_list, v_trimmed_list)]
    u_dot_resampled_list = [PchipInterpolator(t_angle, u_dot)(t_resampled) for t_angle, u_dot in zip(t_angle_trimmed_list, u_dot_trimmed_list)]
    v_dot_resampled_list = [PchipInterpolator(t_angle, v_dot)(t_resampled) for t_angle, v_dot in zip(t_angle_trimmed_list, v_dot_trimmed_list)]
    u_cmd_resampled_list = [PchipInterpolator(t_cmd, u_cmd)(t_resampled) for t_cmd, u_cmd in zip(t_cmd_trimmed_list, u_cmd_trimmed_list)]
    v_cmd_resampled_list = [PchipInterpolator(t_cmd, v_cmd)(t_resampled) for t_cmd, v_cmd in zip(t_cmd_trimmed_list, v_cmd_trimmed_list)]
    
    # Normalize timestamps
    t_resampled = t_resampled - t_resampled[0]
    
    return t_resampled, u_resampled_list, v_resampled_list, u_dot_resampled_list, v_dot_resampled_list, u_cmd_resampled_list, v_cmd_resampled_list

def resample_data_for_same_length_all(t_span:List, t_angle_trimmed: np.ndarray, t_cmd_trimmed: np.ndarray, u_trimmed: List[np.ndarray],
                                        v_trimmed: List[np.ndarray], u_dot_trimmed: List[np.ndarray], v_dot_trimmed: List[np.ndarray],
                                        u_cmd_trimmed: List[np.ndarray], v_cmd_trimmed: List[np.ndarray], t_pressure_state_trimmed: List[np.ndarray],
                                        p0_trimmed: List[np.ndarray], p1_trimmed: List[np.ndarray], p2_trimmed: List[np.ndarray], p3_trimmed: List[np.ndarray],
                                        t_pressure_cmd_trimmed: List[np.ndarray], p_cmd0_trimmed: List[np.ndarray], p_cmd1_trimmed: List[np.ndarray],
                                        p_cmd2_trimmed: List[np.ndarray], p_cmd3_trimmed: List[np.ndarray], rate:float) -> tuple:
    """
    Resamples data to be the same length.
    :param t_angle_trimmed: numpy array representing trimmed timestamps for angle measurements.
    :param t_cmd_trimmed: numpy array representing trimmed timestamps for commands.
    :param u_trimmed: numpy array representing trimmed angle measurements for joint u.
    :param v_trimmed: numpy array representing trimmed angle measurements for joint v.
    :param u_dot_trimmed: numpy array representing trimmed angular velocities for joint u.
    :param v_dot_trimmed: numpy array representing trimmed angular velocities for joint v.
    :param u_cmd_trimmed: numpy array representing trimmed commands for joint u.
    :param v_cmd_trimmed: numpy array representing trimmed commands for joint v.
    :param t_pressure_state_trimmed: numpy array representing trimmed timestamps for pressure measurements.
    :param p0_trimmed: numpy array representing trimmed pressure measurements for joint 0.
    :param p1_trimmed: numpy array representing trimmed pressure measurements for joint 1.
    :param p2_trimmed: numpy array representing trimmed pressure measurements for joint 2.
    :param p3_trimmed: numpy array representing trimmed pressure measurements for joint 3.
    :param t_pressure_cmd_trimmed: numpy array representing trimmed timestamps for pressure commands.
    :param p_cmd0_trimmed: numpy array representing trimmed pressure commands for joint 0.
    :param p_cmd1_trimmed: numpy array representing trimmed pressure commands for joint 1.
    :param p_cmd2_trimmed: numpy array representing trimmed pressure commands for joint 2.
    :param p_cmd3_trimmed: numpy array representing trimmed pressure commands for joint 3.
    :param rate: float representing the rate at which to resample the data.
    :return: Tuple of resampled timestamps and resampled data for angle measurements, velocities, and commands, and pressure measurements and commands.
    """
    # Resample the timestamps to be the same length
    t_resampled = np.arange(t_span[0], t_span[1], rate)

    # Interpolate data to the resampled timestamps
    u_resampled_list = [PchipInterpolator(t_angle, u)(t_resampled) for t_angle, u in zip(t_angle_trimmed, u_trimmed)]
    v_resampled_list = [PchipInterpolator(t_angle, v)(t_resampled) for t_angle, v in zip(t_angle_trimmed, v_trimmed)]
    u_dot_resampled_list = [PchipInterpolator(t_angle, u_dot)(t_resampled) for t_angle, u_dot in zip(t_angle_trimmed, u_dot_trimmed)]
    v_dot_resampled_list = [PchipInterpolator(t_angle, v_dot)(t_resampled) for t_angle, v_dot in zip(t_angle_trimmed, v_dot_trimmed)]
    u_cmd_resampled_list = [PchipInterpolator(t_cmd, u_cmd)(t_resampled) for t_cmd, u_cmd in zip(t_cmd_trimmed, u_cmd_trimmed)]
    v_cmd_resampled_list = [PchipInterpolator(t_cmd, v_cmd)(t_resampled) for t_cmd, v_cmd in zip(t_cmd_trimmed, v_cmd_trimmed)]
    p0_resampled_list = [PchipInterpolator(t_pressure_state, p0)(t_resampled) for t_pressure_state, p0 in zip(t_pressure_state_trimmed, p0_trimmed)]
    p1_resampled_list = [PchipInterpolator(t_pressure_state, p1)(t_resampled) for t_pressure_state, p1 in zip(t_pressure_state_trimmed, p1_trimmed)]
    p2_resampled_list = [PchipInterpolator(t_pressure_state, p2)(t_resampled) for t_pressure_state, p2 in zip(t_pressure_state_trimmed, p2_trimmed)]
    p3_resampled_list = [PchipInterpolator(t_pressure_state, p3)(t_resampled) for t_pressure_state, p3 in zip(t_pressure_state_trimmed, p3_trimmed)]
    p_cmd0_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd0)(t_resampled) for t_pressure_cmd, p_cmd0 in zip(t_pressure_cmd_trimmed, p_cmd0_trimmed)]
    p_cmd1_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd1)(t_resampled) for t_pressure_cmd, p_cmd1 in zip(t_pressure_cmd_trimmed, p_cmd1_trimmed)]
    p_cmd2_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd2)(t_resampled) for t_pressure_cmd, p_cmd2 in zip(t_pressure_cmd_trimmed, p_cmd2_trimmed)]
    p_cmd3_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd3)(t_resampled) for t_pressure_cmd, p_cmd3 in zip(t_pressure_cmd_trimmed, p_cmd3_trimmed)]

    # Normalize timestamps
    t_resampled = t_resampled - t_resampled[0]

    return t_resampled, u_resampled_list, v_resampled_list, u_dot_resampled_list, v_dot_resampled_list, u_cmd_resampled_list, v_cmd_resampled_list, p0_resampled_list, p1_resampled_list, p2_resampled_list, p3_resampled_list, p_cmd0_resampled_list, p_cmd1_resampled_list, p_cmd2_resampled_list, p_cmd3_resampled_list

def resample_data_for_same_length_data_collection(t_span:List, t_angle_trimmed: np.ndarray, u_trimmed: List[np.ndarray], v_trimmed: List[np.ndarray], 
                                      u_dot_trimmed: List[np.ndarray], v_dot_trimmed: List[np.ndarray], t_pressure_state_trimmed: List[np.ndarray],
                                        p0_trimmed: List[np.ndarray], p1_trimmed: List[np.ndarray], p2_trimmed: List[np.ndarray], p3_trimmed: List[np.ndarray],
                                        t_pressure_cmd_trimmed: List[np.ndarray], p_cmd0_trimmed: List[np.ndarray], p_cmd1_trimmed: List[np.ndarray],
                                        p_cmd2_trimmed: List[np.ndarray], p_cmd3_trimmed: List[np.ndarray], rate:float) -> tuple:
    """
    Resamples data to be the same length.
    :param t_angle_trimmed: numpy array representing trimmed timestamps for angle measurements.
    :param u_trimmed: numpy array representing trimmed angle measurements for joint u.
    :param v_trimmed: numpy array representing trimmed angle measurements for joint v.
    :param u_dot_trimmed: numpy array representing trimmed angular velocities for joint u.
    :param v_dot_trimmed: numpy array representing trimmed angular velocities for joint v.
    :param t_pressure_state_trimmed: numpy array representing trimmed timestamps for pressure measurements.
    :param p0_trimmed: numpy array representing trimmed pressure measurements for joint 0.
    :param p1_trimmed: numpy array representing trimmed pressure measurements for joint 1.
    :param p2_trimmed: numpy array representing trimmed pressure measurements for joint 2.
    :param p3_trimmed: numpy array representing trimmed pressure measurements for joint 3.
    :param t_pressure_cmd_trimmed: numpy array representing trimmed timestamps for pressure commands.
    :param p_cmd0_trimmed: numpy array representing trimmed pressure commands for joint 0.
    :param p_cmd1_trimmed: numpy array representing trimmed pressure commands for joint 1.
    :param p_cmd2_trimmed: numpy array representing trimmed pressure commands for joint 2.
    :param p_cmd3_trimmed: numpy array representing trimmed pressure commands for joint 3.
    :param rate: float: Resampling rate.
    :return: Tuple of resampled timestamps and resampled data for angle measurements, velocities, and commands, and pressure measurements and commands.
    """
    # Resample the timestamps to be the same length 
    t_resampled = np.arange(t_span[0], t_span[1], rate)

    # Interpolate data to the resampled timestamps
    u_resampled_list = [PchipInterpolator(t_angle, u)(t_resampled) for t_angle, u in zip(t_angle_trimmed, u_trimmed)]
    v_resampled_list = [PchipInterpolator(t_angle, v)(t_resampled) for t_angle, v in zip(t_angle_trimmed, v_trimmed)]
    u_dot_resampled_list = [PchipInterpolator(t_angle, u_dot)(t_resampled) for t_angle, u_dot in zip(t_angle_trimmed, u_dot_trimmed)]
    v_dot_resampled_list = [PchipInterpolator(t_angle, v_dot)(t_resampled) for t_angle, v_dot in zip(t_angle_trimmed, v_dot_trimmed)]

    p0_resampled_list = [PchipInterpolator(t_pressure_state, p0)(t_resampled) for t_pressure_state, p0 in zip(t_pressure_state_trimmed, p0_trimmed)]
    p1_resampled_list = [PchipInterpolator(t_pressure_state, p1)(t_resampled) for t_pressure_state, p1 in zip(t_pressure_state_trimmed, p1_trimmed)]
    p2_resampled_list = [PchipInterpolator(t_pressure_state, p2)(t_resampled) for t_pressure_state, p2 in zip(t_pressure_state_trimmed, p2_trimmed)]
    p3_resampled_list = [PchipInterpolator(t_pressure_state, p3)(t_resampled) for t_pressure_state, p3 in zip(t_pressure_state_trimmed, p3_trimmed)]
    p_cmd0_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd0)(t_resampled) for t_pressure_cmd, p_cmd0 in zip(t_pressure_cmd_trimmed, p_cmd0_trimmed)]
    p_cmd1_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd1)(t_resampled) for t_pressure_cmd, p_cmd1 in zip(t_pressure_cmd_trimmed, p_cmd1_trimmed)]
    p_cmd2_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd2)(t_resampled) for t_pressure_cmd, p_cmd2 in zip(t_pressure_cmd_trimmed, p_cmd2_trimmed)]
    p_cmd3_resampled_list = [PchipInterpolator(t_pressure_cmd, p_cmd3)(t_resampled) for t_pressure_cmd, p_cmd3 in zip(t_pressure_cmd_trimmed, p_cmd3_trimmed)]

    # Normalize timestamps
    t_resampled = t_resampled - t_resampled[0]

    return t_resampled, u_resampled_list, v_resampled_list, u_dot_resampled_list, v_dot_resampled_list, p0_resampled_list, p1_resampled_list, p2_resampled_list, p3_resampled_list, p_cmd0_resampled_list, p_cmd1_resampled_list, p_cmd2_resampled_list, p_cmd3_resampled_list

def save_csv(save_path:str, resampled_tspan:np.ndarray, u_resampled_trimmed:List[np.ndarray], v_resampled_trimmed:List[np.ndarray],
                u_dot_resampled_trimmed:List[np.ndarray], v_dot_resampled_trimmed:List[np.ndarray], 
                p0_resampled_trimmed:List[np.ndarray], p1_resampled_trimmed:List[np.ndarray], p2_resampled_trimmed:List[np.ndarray], p3_resampled_trimmed:List[np.ndarray],
                p_cmd0_resampled_trimmed:List[np.ndarray], p_cmd1_resampled_trimmed:List[np.ndarray], p_cmd2_resampled_trimmed:List[np.ndarray], p_cmd3_resampled_trimmed:List[np.ndarray],
                u_cmd_resampled_trimmed:List[np.ndarray]=None, v_cmd_resampled_trimmed:List[np.ndarray]=None, joint_iae:np.ndarray=None):
    """
    Saves the resampled data to a csv file.
    :param save_path: str: Path to save the csv file.
    :param resampled_tspan: np.ndarray: Resampled timestamps.
    :param u_resampled_trimmed: np.ndarray: U angle measurements, list of numpy arrays.
    :param v_resampled_trimmed: np.ndarray: V angle measurements, list of numpy arrays.
    :param u_dot_resampled_trimmed: np.ndarray: U angular velocities, list of numpy arrays.
    :param v_dot_resampled_trimmed: np.ndarray: V angular velocities, list of numpy arrays.
    :param p0_resampled_trimmed: np.ndarray: Pressure measurements for bellows chamber 0, list of numpy arrays.
    :param p1_resampled_trimmed: np.ndarray: Pressure measurements for bellows chamber 1, list of numpy arrays.
    :param p2_resampled_trimmed: np.ndarray: Pressure measurements for bellows chamber 2, list of numpy arrays.
    :param p3_resampled_trimmed: np.ndarray: Pressure measurements for bellows chamber 3, list of numpy arrays.
    :param p_cmd0_resampled_trimmed: np.ndarray: Pressure commands for bellows chamber 0, list of numpy arrays.
    :param p_cmd1_resampled_trimmed: np.ndarray: Pressure commands for bellows chamber 1, list of numpy arrays.
    :param p_cmd2_resampled_trimmed: np.ndarray: Pressure commands for bellows chamber 2, list of numpy arrays.
    :param p_cmd3_resampled_trimmed: np.ndarray: Pressure commands for bellows chamber 3, list of numpy arrays.
    :param u_cmd_resampled_trimmed: np.ndarray: U angle commands, list of numpy arrays.
    :param v_cmd_resampled_trimmed: np.ndarray: V angle commands, list of numpy arrays.
    :param joint_iae: np.ndarray: Integral absolute error score.

    :return: None
    """

    save_dict =        {
            "time": resampled_tspan,
            "u": u_resampled_trimmed,
            "v": v_resampled_trimmed,
            "u_dot": u_dot_resampled_trimmed,
            "v_dot": v_dot_resampled_trimmed,
            "p0": p0_resampled_trimmed,
            "p1": p1_resampled_trimmed,
            "p2": p2_resampled_trimmed,
            "p3": p3_resampled_trimmed,
            "p_cmd0": p_cmd0_resampled_trimmed,
            "p_cmd1": p_cmd1_resampled_trimmed,
            "p_cmd2": p_cmd2_resampled_trimmed,
            "p_cmd3": p_cmd3_resampled_trimmed,
        }
    
    if u_cmd_resampled_trimmed is not None:
        save_dict["u_cmd"] = u_cmd_resampled_trimmed
        save_dict["v_cmd"] = v_cmd_resampled_trimmed
    if joint_iae is not None:
        save_dict["joint_iae"] = joint_iae

    data = pd.DataFrame(save_dict)

    try:
        os.mkdir("/".join(save_path.split("/")[:-1]))
    except:
        pass
    data.to_csv(save_path + ".csv", index=False)

def get_data_from_csv(csv_path:str):
    df = pd.read_csv(csv_path)

    t = df["time"].to_numpy()
    u = df['u'].to_numpy()
    v = df['v'].to_numpy()
    udot = df['u_dot'].to_numpy()
    vdot = df['v_dot'].to_numpy()
    p0 = df['p0'].to_numpy()
    p1 = df['p1'].to_numpy()
    p2 = df['p2'].to_numpy()
    p3 = df['p3'].to_numpy()
    pc0 = df['p_cmd0'].to_numpy()
    pc1 = df['p_cmd1'].to_numpy()
    pc2 = df['p_cmd2'].to_numpy()
    pc3 = df['p_cmd3'].to_numpy()

    return np.array([t, p0, p1, p2, p3, udot, vdot, u, v, pc0, pc1, pc2, pc3]).T

def get_data_from_csv_control(csv_path:str):
    df = pd.read_csv(csv_path)

    t = df["time"].to_numpy()
    u = df['u'].to_numpy()
    v = df['v'].to_numpy()
    udot = df['u_dot'].to_numpy()
    vdot = df['v_dot'].to_numpy()
    p0 = df['p0'].to_numpy()
    p1 = df['p1'].to_numpy()
    p2 = df['p2'].to_numpy()
    p3 = df['p3'].to_numpy()
    pc0 = df['p_cmd0'].to_numpy()
    pc1 = df['p_cmd1'].to_numpy()
    pc2 = df['p_cmd2'].to_numpy()
    pc3 = df['p_cmd3'].to_numpy()
    u_cmd = df['u_cmd'].to_numpy()
    v_cmd = df['v_cmd'].to_numpy()

    return np.array([t, p0, p1, p2, p3, udot, vdot, u, v, pc0, pc1, pc2, pc3, u_cmd, v_cmd]).T

def calculate_iae(u_resampled_trimmed:List[np.ndarray], u_cmd_resampled_trimmed:List[np.ndarray],
                    v_resampled_trimmed:List[np.ndarray], v_cmd_resampled_trimmed:List[np.ndarray]) -> np.ndarray:
    """
    Calculates the integral absolute error score.
    :param u_resampled_trimmed: np.ndarray: U angle measurements, list of numpy arrays.
    :param u_cmd_resampled_trimmed: np.ndarray: U angle commands, list of numpy arrays.
    :param v_resampled_trimmed: np.ndarray: V angle measurements, list of numpy arrays.
    :param v_cmd_resampled_trimmed: np.ndarray: V angle commands, list of numpy arrays.

    :return: np.ndarray: Integral absolute error score.
    """

    joint_iae = 0
    for i in range(len(u_resampled_trimmed)):
        joint_iae += (np.sum(np.abs(u_resampled_trimmed[i] - u_cmd_resampled_trimmed[i])) 
        + np.sum(np.abs(v_resampled_trimmed[i] - v_cmd_resampled_trimmed[i])))

    return joint_iae

def bag_to_csv_control(b:bagreader, save_path:str, topic_names:List[str], rate:float=0.01):
    """
    Extracts data from a rosbag and saves it to a csv file.
    :param b: bagreader: rosbag file.
    :param save_path: str: Path to save the csv file.
    :param topic_names: List[str]: List of topic names to extract data from.
    :return: None
    """

    t_angles, us, vs, u_dots, v_dots = get_angle_data(b, topic_names[0])
    t_angle_commands, u_cmds, v_cmds = get_joint_command(b, topic_names[1])
    t_pressures, p0s, p1s, p2s, p3s = get_pressure_data(b, topic_names[2])
    t_pressure_cmds, p0_cmds, p1_cmds, p2_cmds, p3_cmds = get_pressure_cmd(b, topic_names[3])

    max_t = [t_angle[0] for t_angle in t_angles] + [t_pressure[0] for t_pressure in t_pressures] + [t_pressure_cmd[0] for t_pressure_cmd in t_pressure_cmds]
    min_t = [t_angle[-1] for t_angle in t_angles] + [t_pressure[-1] for t_pressure in t_pressures] + [t_pressure_cmd[-1] for t_pressure_cmd in t_pressure_cmds]
    trimmed_tspan = [max(max_t), min(min_t)]

    t_angle_trimmed_list, u_trimmed_list, v_trimmed_list, u_dot_trimmed_list, \
        v_dot_trimmed_list, u_cmd_trimmed_list, v_cmd_trimmed_list, t_cmd_trimmed_list = get_trimmed_data_joint_command(trimmed_tspan, t_angles, t_angle_commands, us, vs, u_dots, v_dots, u_cmds, v_cmds)

    t_pressure_trimmed_list, p0_trimmed_list, p1_trimmed_list, p2_trimmed_list, p3_trimmed_list, \
        t_pressure_cmd_trimmed_list, p_cmd0_trimmed_list, p_cmd1_trimmed_list, p_cmd2_trimmed_list, p_cmd3_trimmed_list = get_trimmed_data_pressure(trimmed_tspan, t_pressures, p0s, p1s, p2s, p3s, t_pressure_cmds, p0_cmds, p1_cmds, p2_cmds, p3_cmds)

    # resample all the data
    t_resampled, u_resampled_list, v_resampled_list, u_dot_resampled_list, v_dot_resampled_list, u_cmd_resampled_list, v_cmd_resampled_list, p0_resampled_list, \
        p1_resampled_list, p2_resampled_list, p3_resampled_list, p_cmd0_resampled_list, p_cmd1_resampled_list, p_cmd2_resampled_list, p_cmd3_resampled_list = resample_data_for_same_length_all(trimmed_tspan, t_angle_trimmed_list, t_cmd_trimmed_list, u_trimmed_list, v_trimmed_list, u_dot_trimmed_list, v_dot_trimmed_list, u_cmd_trimmed_list, v_cmd_trimmed_list, t_pressure_trimmed_list, p0_trimmed_list, p1_trimmed_list, p2_trimmed_list, p3_trimmed_list, t_pressure_cmd_trimmed_list, p_cmd0_trimmed_list, p_cmd1_trimmed_list, p_cmd2_trimmed_list, p_cmd3_trimmed_list, rate)
    
    # calculate iae
    joint_iae = calculate_iae(u_resampled_list, u_cmd_resampled_list, v_resampled_list, v_cmd_resampled_list)

    # save data
    for i in range(len(us)):
        save_csv(save_path+f"_{i}", t_resampled, u_resampled_list[i], v_resampled_list[i], u_dot_resampled_list[i], v_dot_resampled_list[i], p0_resampled_list[i], p1_resampled_list[i], p2_resampled_list[i], p3_resampled_list[i], p_cmd0_resampled_list[i], p_cmd1_resampled_list[i], p_cmd2_resampled_list[i], p_cmd3_resampled_list[i], u_cmd_resampled_list[i], v_cmd_resampled_list[i], joint_iae)

    return joint_iae

def bag_to_csv_data_collection(b:bagreader, save_path:str, topic_names:List[str], rate:float=0.01):
    """
    Extracts data from a rosbag and saves it to a csv file.
    :param b: bagreader: rosbag file.
    :param save_path: str: Path to save the csv file.
    :param topic_names: List[str]: List of topic names to extract data from.
    :param rate: float: Resampling rate.
    :return: None
    """

    t_angles, us, vs, u_dots, v_dots = get_angle_data(b, topic_names[0])
    t_pressures, p0s, p1s, p2s, p3s = get_pressure_data(b, topic_names[1])
    t_pressure_cmds, p0_cmds, p1_cmds, p2_cmds, p3_cmds = get_pressure_cmd(b, topic_names[2])

    max_t = [t_angle[0] for t_angle in t_angles] + [t_pressure[0] for t_pressure in t_pressures] + [t_pressure_cmd[0] for t_pressure_cmd in t_pressure_cmds]
    min_t = [t_angle[-1] for t_angle in t_angles] + [t_pressure[-1] for t_pressure in t_pressures] + [t_pressure_cmd[-1] for t_pressure_cmd in t_pressure_cmds]
    trimmed_tspan = [max(max_t), min(min_t)]

    t_angle_trimmed_list, u_trimmed_list, v_trimmed_list, u_dot_trimmed_list, v_dot_trimmed_list = get_trimmed_data_collection(trimmed_tspan, t_angles, us, vs, u_dots, v_dots)
    
    t_pressure_trimmed_list, p0_trimmed_list, p1_trimmed_list, p2_trimmed_list, p3_trimmed_list, \
        t_pressure_cmd_trimmed_list, p_cmd0_trimmed_list, p_cmd1_trimmed_list, p_cmd2_trimmed_list, p_cmd3_trimmed_list = get_trimmed_data_pressure(trimmed_tspan, t_pressures, p0s, p1s, p2s, p3s, t_pressure_cmds, p0_cmds, p1_cmds, p2_cmds, p3_cmds)

    # resample all the data
    t_resampled, u_resampled_list, v_resampled_list, u_dot_resampled_list, v_dot_resampled_list, p0_resampled_list, p1_resampled_list, p2_resampled_list,\
          p3_resampled_list, p_cmd0_resampled_list, p_cmd1_resampled_list, p_cmd2_resampled_list, p_cmd3_resampled_list = resample_data_for_same_length_data_collection(trimmed_tspan, t_angle_trimmed_list, u_trimmed_list, v_trimmed_list, u_dot_trimmed_list, v_dot_trimmed_list, t_pressure_trimmed_list, p0_trimmed_list, p1_trimmed_list, p2_trimmed_list, p3_trimmed_list, t_pressure_cmd_trimmed_list, p_cmd0_trimmed_list, p_cmd1_trimmed_list, p_cmd2_trimmed_list, p_cmd3_trimmed_list, rate)

    # save data
    paths = []
    for i in range(len(us)):
        save_csv(save_path+f"/not_smooth_joint{i}", t_resampled, u_resampled_list[i], v_resampled_list[i], u_dot_resampled_list[i], v_dot_resampled_list[i], p0_resampled_list[i], p1_resampled_list[i], p2_resampled_list[i], p3_resampled_list[i], p_cmd0_resampled_list[i], p_cmd1_resampled_list[i], p_cmd2_resampled_list[i], p_cmd3_resampled_list[i])
        # save_csv(save_path+f"/smooth_joint{i}", t_resampled, u_hat_list[i], v_hat_list[i], u_dot_hat_list[i], v_dot_hat_list[i], p0_hat_list[i], p1_hat_list[i], p2_hat_list[i], p3_hat_list[i], p_cmd0_resampled_list[i], p_cmd1_resampled_list[i], p_cmd2_resampled_list[i], p_cmd3_resampled_list[i])
        paths.append(save_path+f"/joint{i}.csv")
    return paths