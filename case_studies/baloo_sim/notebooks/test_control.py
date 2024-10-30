from moldy.case_studies.baloo_sim.nempc_baloo_sim import nempc_baloo_setup
from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim
from moldy.validation.control.control_test import ControlTest
import numpy as np

tester = ControlTest(ground_truth=BalooSim("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_baloo.xml"), 
                    nempc_setup=nempc_baloo_setup,
                    sim_seconds=100.0,
                    num_step_commands=10,
                    xgoal_path="/home/daniel/Documents/data/xfer_learning/sim_to_sim_baloo/baloo_paper_joint_commands.npy",
                    states_of_interest=[18, 19, 20, 21, 22, 23],
                    )

horizon = 40
numSims = 1000
numParents = 400
numStrangers = 100
numKnotPoints = 1
selection_method = "tournament"
tournament_size = 5
crossover_method = "knot_point"
mutation_probability = 0.1
Q = np.diag([0,0,0,0,0,0,0,0,0,0,0,0,
            # 0.21, 0.21, 0.21, 0.21, 0.21, 0.21,  # velocity weights
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # velocity weights
            # 0, 0.0, 0.0, 0.0, 0.0, 0.0,  # velocity weights
            50, 50, 50, 50, 50, 50               # position weights
            ])  
R = 1e-4 * np.diag([1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0,])
ctrl_dt = 0.02
nempc_params = {"horizon": horizon,
                "numSims": numSims,
                "numParents": numParents,
                "numStrangers": numStrangers,
                "numKnotPoints": numKnotPoints,
                "selection_method": selection_method,
                "tournament_size": tournament_size,
                "crossover_method": crossover_method,
                "mutation_probability": mutation_probability,
                "Q": Q,
                "R": R,
                }
def setup_wrapper(trial_dir=None, x0=None):
    return nempc_baloo_setup(trial_dir=trial_dir, x0=x0, horizon=horizon, numSims=numSims, numParents=numParents, numStrangers=numStrangers, numKnotPoints=numKnotPoints, Q=Q, R=R, ctrl_dt=ctrl_dt)


tester.run_trials(
    "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/test_results/",
                                "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/",
                run_analytical=False,
                nempc_setup=setup_wrapper,
                nempc_params=nempc_params,
                ctrl_dt=ctrl_dt,
                test_name="ALL_BEST_MODELS_FINAL",
                )

# tester.run_trials("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/test_results/",
#                 "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/250KMODELS/",
#                 run_analytical=False,
#                 nempc_setup=setup_wrapper,
#                 nempc_params=nempc_params,
#                 ctrl_dt=ctrl_dt,
#                 test_name="testing_250KMODELS",
#                 )

# tester.run_trials("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/test_results/",
#                 "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/500KMODELS/",
#                 run_analytical=False,
#                 nempc_setup=setup_wrapper,
#                 nempc_params=nempc_params,
#                 ctrl_dt=ctrl_dt,
#                 test_name="testing_500KMODELS",
#                 )

# tester.run_trials("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/test_results/",
#                 "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/1MMODELS/",
#                 run_analytical=False,
#                 nempc_setup=setup_wrapper,
#                 nempc_params=nempc_params,
#                 ctrl_dt=ctrl_dt,
#                 test_name="testing_1MMODELS",
#                 )

# tester.run_trials("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/test_results/",
#                 "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/1.5MMODELS/",
#                 run_analytical=False,
#                 nempc_setup=setup_wrapper,
#                 nempc_params=nempc_params,
#                 ctrl_dt=ctrl_dt,
#                 test_name="testing_1.5MMODELS",
#                 )
