import numpy as np
import torch
from typing import Tuple, Union
from copy import deepcopy


from moldy.validation.control.nempc.NonlinearEMPC import NonlinearEMPC as NEMPC
from moldy.validation.control.nempc.nempc_base import (
    nempc_conditions_setup,
    nempc_run,
)
from moldy.model.Model import Model

from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim
from moldy.case_studies.baloo_sim.learned_model_baloo_sim import LearnedModel_BalooSim

def nempc_baloo_setup(
    trial_dir: str=None,
    ctrl_dt:float=0.01,
    horizon:int=50,
    numSims:int=500,
    numParents:int=200,
    numStrangers:int=50,
    numKnotPoints:int=1,
    selection_method:str="tournament",
    tournament_size:int=5,
    crossover_method:str="knot_point",
    mutation_probability:float=0.1,
    Q:np.ndarray=None,
    R:np.ndarray=None,
    x0:np.ndarray=None,
    differential_pressures=False
    ) -> Tuple[NEMPC, Model]:
    """
    Set up the NEMPC controller for the grub.
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
        Q = np.diag([0,0,0,0,0,0,0,0,0,0,0,0,
                    0.21, 0.21, 0.21, 0.21, 0.21, 0.21,  # velocity weights
                    # 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # velocity weights
                    50, 50, 50, 50, 50, 50               # position weights
                    ])  
    if R is None:
        R = 1e-5 * np.diag([1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0,])

    if trial_dir is None:
        system = BalooSim(numSims=numSims)
        use_gpu = False
        if not differential_pressures:
            umin = system.uMin
            umax = system.uMax
        else:
            umin = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
            umax = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    else:
        use_gpu = True
        system = LearnedModel_BalooSim(trial_dir=trial_dir)
        Q = torch.from_numpy(Q).float().cuda()
        R = torch.from_numpy(R).float().cuda()
        if not differential_pressures:
            umin = system.uMin.detach().cpu().numpy()
            umax = system.uMax.detach().cpu().numpy()
        else:
            umin = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
            umax = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


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
            # Rx = np.matmul((u-prev_u) ** 2.0, R)
            cost = np.sum(Qx, axis=1)

        return cost
    
    ground_truth = BalooSim(XML_PATH="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_baloo.xml")
    _x0, xgoal = nempc_conditions_setup(ground_truth)
    _u0 = _x0[:, :ground_truth.numInputs]

    if x0 is None:
        x0 = _x0

    ground_truth.set_state(x0, -1)
    for i in range(100):
        ground_truth.forward_simulate_dt(x0, _u0, 0.01)
    x0 = ground_truth.get_state(i=0)

    for i in range(len(system.data)):
        system.data[i] = deepcopy(ground_truth.data[0])
    
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
    controller, system, ground_truth, x0, xgoal = nempc_baloo_setup(
        # trial_dir=None,
        trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/daniels_smooth_L1_weighted_mult1_10_mult2_5",
        # differential_pressures=True
    )

    iae_score, x_history, u_history, t_history = nempc_run(
        controller,
        system,
        ground_truth,
        x0,
        xgoal,
        sim_length=6.0,
        mutationNoise=5.0,
        sim_dt=0.01,
        using_mujoco=True,
        visualize_horizon=True,
        # visualize=True,
        print_stats=True,
        plot=True,
    )
