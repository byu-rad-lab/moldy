import numpy as np
import time
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Tuple
import os

from moldy.validation.control.nempc.NonlinearEMPC import NonlinearEMPC as NEMPC
from moldy.model.Model import Model

def nempc_conditions_setup(system: Model, n:int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random initial conditions, commands, and goal states for the NEMPC controller.
    :param system: Model: Analytical or Learned Class that inherits from Model.py
    :param n: int: Number of initial conditions to generate
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0: np.ndarray: Initial conditions, size (n, numStates)
        xgoal: np.ndarray: Goal states, size (n, numStates)
    """
    x0 = system.generate_random_state(initial_conditions=True, n=n)
    xgoal = system.generate_random_xgoal(n=n)
    return x0, xgoal

def nempc_run(
    controller: NEMPC,
    trial_model: Model,
    ground_truth: Model,
    x0: np.ndarray,
    xgoal: np.ndarray,
    sim_length:float=2.0,
    sim_dt:float=0.01,
    mutationNoise:float=0.99,
    **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the NEMPC controller on the given model.
    :param controller: NEMPC: NEMPC controller
    :param trial_model: Model: Analytical or Learned Class that inherits from Model.py
    :param ground_truth: Model: Analytical Class that inherits from Model.py
    :param x0: np.ndarray: Initial conditions, size (1, numStates)
    :param xgoal: np.ndarray: Goal states, size (1, numStates)
    :param sim_length: float: Length of time to simulate the system
    :param sim_dt: float: Timestep to use for simulation
    :param mutationNoise: float: Mutation noise to use for NEMPC
    :param kwargs: dict: Optional arguments
        :param kwargs["visualize_horizon"]: bool: Visualize the NEMPC horizon
        :param kwargs["plot"]: bool: Plot the NEMPC run
        :param kwargs["visualize_system"]: bool: Visualize the system
        :param kwargs["make_gif"]: bool: Make a gif of the control trajectory
        :param kwargs["print_stats"]: bool: whether or not to print the stats from running control
        :param kwargs["using_mujoco"] bool: whether or not the control model is using mujoco

    :return: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        iae_score: np.ndarray: Integral absolute error score, size (1, numStates)
        x_history: np.ndarray: History of states, size (sim_steps, numStates)
        u_history: np.ndarray: History of commands, size (sim_steps, numInputs)
        t_history: np.ndarray: History of computation times, size (sim_steps, 1)
    """

    _visualize_horizon = kwargs.get("visualize_horizon", False)
    plot = kwargs.get("plot", False)
    visualize_system = kwargs.get("visualize_system", False)
    make_gif = kwargs.get("make_gif", False)
    print_stats = kwargs.get("print_stats", False)
    using_mujoco = kwargs.get("using_mujoco", False)

    sim_steps = int(sim_length / controller.dt)
    x_history = np.zeros((sim_steps, controller.numStates))
    u_history = np.zeros((sim_steps, controller.numInputs))
    t_history = np.zeros((sim_steps, 1))
    x = deepcopy(x0)

    if _visualize_horizon:
        plt.ion()

    if make_gif:
        os.makedirs("nempc_gif", exist_ok=True)
        os.makedirs("nempc_gif/visualization", exist_ok=True)
        os.makedirs("nempc_gif/performance", exist_ok=True)
        visualization_path = "nempc_gif/visualization"
        performance_path = "nempc_gif/performance"

    if xgoal.shape[0] == 1:
        xgoal = np.tile(xgoal, (sim_steps, 1))

    with torch.no_grad():
        for i in range(sim_steps):
            start = time.time()
            u_nempc, path = controller.solve_for_next_u(
                x,
                xgoal[i],
                ulast=u_history[i - 1],
                ugoal=np.zeros((1, controller.numInputs)),
                mutation_noise=mutationNoise,
                base_model=(None if not using_mujoco else ground_truth.data)
            )

            end = time.time()
            u = u_nempc[0, :]
            
            x_history[i] = x
            u_history[i] = u
            t_history[i] = end - start

            for j in range(int(controller.dt / sim_dt)):
                x = ground_truth.forward_simulate_dt(x, u, sim_dt)
                # x = ground_truth.forward_simulate_dt(x, ground_truth.convert_diff_to_command(u), sim_dt)

            if _visualize_horizon:
                ground_truth.visualize_horizon(
                                                x_history,
                                                xgoal[i],
                                                path,
                                                controller.horizon,
                                                sim_steps,
                                                i,
                                            )
                if make_gif:
                    plt.savefig(f"{performance_path}/{i:04d}.png")
                    plt.close()

            if visualize_system:
                fig = ground_truth.visualize(x)

                if make_gif:
                    fig.savefig(f"{visualization_path}/{i:04d}.png")
                    plt.close(fig)

    iae_score = np.sum(np.abs(x_history - xgoal), axis=0).reshape((1, controller.numStates))

    if plot:
        trial_model.plot_history(x_history, u_history, xgoal)

    if make_gif:
        print("Saving GIFs")
        from PIL import Image
        visualization_frames = [Image.open(f"{visualization_path}/{i:04d}.png") for i in range(sim_steps)]
        performance_frames = [Image.open(f"{performance_path}/{i:04d}.png") for i in range(sim_steps)]

        visualization_frames[0].save("nempc_gif/visualization.gif", save_all=True, append_images=visualization_frames[1:], duration=60, loop=0)
        performance_frames[0].save("nempc_gif/performance.gif", save_all=True, append_images=performance_frames[1:], duration=60, loop=0)

    if print_stats:
        print(f"IAE Score: {iae_score}")
        print(f"Average Computation Time: {np.mean(t_history)}")

    return iae_score, x_history, u_history, t_history
