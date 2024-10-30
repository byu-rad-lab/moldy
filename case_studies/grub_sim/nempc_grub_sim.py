import numpy as np
import torch
from typing import Tuple, Union

from moldy.validation.control.nempc.NonlinearEMPC import NonlinearEMPC as NEMPC
from moldy.validation.control.nempc.nempc_base import nempc_conditions_setup, nempc_run
from moldy.model.Model import Model

from moldy.case_studies.grub_sim.model_grub_sim import GrubSim
from moldy.case_studies.grub_sim.learnedModel_grub_sim import LearnedModel_GrubSim

def nempc_grub_setup(
    trial_dir: str=None,
    ctrl_dt:float=0.01,
    horizon:int=50,
    numSims:int=1000,
    numParents:int=500,
    numStrangers:int=200,
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
    Set up the NEMPC controller for the grub. If trial_dir is None, the analytical model is used. Otherwise, the learned model is used.
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
        Q = 1.0 * np.diag([0, 0, 0, 0, 0.21, 0.21, 50.0, 50.0])

    if R is None:
        R = 0.0 * np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


    if trial_dir is None:
        # system = GrubSim(**grub_params)
        system = GrubSim()
        use_gpu = False
        umin = system.uMin
        umax = system.uMax
        params = None
    else:
        use_gpu = True
        system = LearnedModel_GrubSim(trial_dir=trial_dir)
        Q = torch.from_numpy(Q).float().cuda()
        R = torch.from_numpy(R).float().cuda()
        umin = system.uMin.detach().cpu().numpy()
        umax = system.uMax.detach().cpu().numpy()
        params = system.model.config.get("grub_params", None)

    if params is None:
        ground_truth = GrubSim()
    else:
        ground_truth = GrubSim(**params)
        
    def CostFunc(x: Union[np.ndarray, torch.Tensor], 
                u: Union[np.ndarray, torch.Tensor], 
                xgoal: Union[np.ndarray, torch.Tensor], 
                ugoal: Union[np.ndarray, torch.Tensor], 
                prev_u: Union[None, np.ndarray, torch.Tensor] = None, 
                final_timestep: bool = False) -> Union[np.ndarray, torch.Tensor]:
        if use_gpu:
            Qx = torch.mm((x - xgoal) ** 2.0, Q)
            Rx = torch.mm((u - prev_u) ** 2.0, R)
            cost = torch.sum(Qx, axis=1) + torch.sum(Rx, axis=1)
        else:
            Qx = np.matmul((x - xgoal) ** 2.0, Q)
            cost = np.sum(Qx, axis=1)
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
    controller, system, ground_truth, x0, xgoal = nempc_grub_setup(
        trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/grub_sim/results/best_models/rmse_4",
        # trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/grub_sim/results/run_logs/base_200K_250E_default_params",
        # trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/grub_sim/results/best_models/mae_33",
        # trial_dir=None,
    )

    iae_score, x_history, u_history, t_history  = nempc_run(
        controller,
        system,
        ground_truth,
        x0,
        xgoal,
        sim_length=2.0,
        visualize_horizon=True,
        plot=True,
        print_stats=True
    )

