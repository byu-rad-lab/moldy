import numpy as np
import torch
from typing import Tuple, Union

from moldy.validation.control.nempc.NonlinearEMPC import NonlinearEMPC as NEMPC
from moldy.validation.control.nempc.nempc_base import (
    nempc_conditions_setup,
    nempc_run,
)
from moldy.model.Model import Model

from moldy.case_studies.inverted_pendulum.model_ip import InvertedPendulum
from moldy.case_studies.inverted_pendulum.learnedModel_ip import LearnedModel_InvertedPendulum

def nempc_ip_setup(
    trial_dir: str=None,
    ctrl_dt:float=0.01,
    horizon:int=75,
    numSims:int=500,
    numParents:int=100,
    numStrangers:int=100,
    numKnotPoints:int=1,
    selection_method:str="tournament",
    tournament_size:int=5,
    crossover_method:str="knot_point",
    mutation_probability:float=0.1,
    Q:np.ndarray=None,
    R:np.ndarray=None,
    x0:np.ndarray=None,
    ) -> Tuple[NEMPC, Model]:
    """
    Set up the NEMPC controller for the inverted pendulum. If trial_dir is None, the analytical model is used. Otherwise, the learned model is used.
    :param trial_dir: The directory containing the learned model. If None, the analytical model is used.
    :param ctrl_dt: The time step of the controller.
    :param horizon: The number of time steps in the controller horizon.
    :param numSims: The number of simulations to run for the control test.
    :param numParents: The number of parents to use in the genetic algorithm.
    :param numStrangers: The number of strangers to use in the genetic algorithm.
    :param numKnotPoints: The number of knot points to use in the genetic algorithm.
    :param selection_method: The selection method to use in the genetic algorithm.
    :param tournament_size: The tournament size to use in the genetic algorithm.
    :param crossover_method: The crossover method to use in the genetic algorithm.
    :param mutation_probability: The mutation probability to use in the genetic algorithm.
    :return: The NEMPC controller and the system model (learned or analytical).
    """
    if Q is None:
        Q = 1.0 * np.diag([0, 1.0])

    ground_truth = InvertedPendulum(length=1.5, mass=0.5, damping=0.05)

    if trial_dir is None:
        system = InvertedPendulum(length=1.5, mass=0.5, damping=0.05)
        use_gpu = False
        umin = system.uMin.squeeze()
        umax = system.uMax.squeeze()
    else:
        system = LearnedModel_InvertedPendulum(trial_dir=trial_dir)
        use_gpu = True
        umin = system.uMin.detach().cpu().numpy().squeeze()
        umax = system.uMax.detach().cpu().numpy().squeeze()
        Q = torch.from_numpy(Q).float().cuda()
    
    wrapAngle = 1
    wrapRange = (-np.pi, np.pi)
    low = wrapRange[0]
    high = wrapRange[1]
    cycle = high - low

    def CostFunc(x: Union[np.ndarray, torch.Tensor], 
                u: Union[np.ndarray, torch.Tensor], 
                xgoal: Union[np.ndarray, torch.Tensor], 
                ugoal: Union[np.ndarray, torch.Tensor], 
                prev_u: Union[None, np.ndarray, torch.Tensor] = None, 
                final_timestep: bool = False) -> Union[np.ndarray, torch.Tensor]:
        x[:, wrapAngle] = (x[:, wrapAngle] + cycle / 2) % cycle + low

        if use_gpu:
            Qx = torch.mm((x - xgoal) ** 2.0, Q)
            cost = torch.sum(Qx, axis=1)
        else:
            Qx = np.matmul((x - xgoal) ** 2.0, Q)
            cost = np.sum(Qx, axis=1)
            
        if final_timestep:
            cost *= 5.0

        return cost

    _x0, xgoal = nempc_conditions_setup(ground_truth)

    if x0 is None:
        x0 = _x0

    controller = NEMPC(
        system.forward_simulate_dt,
        CostFunc,
        numStates=system.numStates,
        numInputs=system.numInputs,
        umin=umin,
        umax=umax,
        horizon=horizon,
        dt=ctrl_dt,
        numKnotPoints=numKnotPoints,
        useGPU=use_gpu,
        selection_method=selection_method,
        tournament_size=tournament_size,
        crossover_method=crossover_method,
        mutation_probability=mutation_probability,
        numSims=numSims,
        numParents=numParents,
        numStrangers=numStrangers,
    )
    return controller, system, ground_truth, x0, xgoal


if __name__ == "__main__":
    controller, system, ground_truth, x0, xgoal = nempc_ip_setup(
        trial_dir="case_studies/inverted_pendulum/results/best_models/opt_cosine_4",
        # trial_dir="moldy/case_studies/inverted_pendulum/results/best_models/opt_rmse_18",
        # trial_dir="moldy/case_studies/inverted_pendulum/results/best_models/same_mse_1",
        # trial_dir=None
    )

    iae_score, x_history, u_history, t_history = nempc_run(
        controller,
        system,
        ground_truth,
        x0,
        xgoal,
        sim_length=15.5,
        mutationNoise=0.63,
        visualize_horizon=True,
        print_stats=True,
        plot=True,
    )