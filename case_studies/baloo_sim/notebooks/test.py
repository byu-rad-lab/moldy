from moldy.validation.prediction.prediction_validation import PredictionValidation

from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim
from moldy.case_studies.baloo_sim.learned_model_baloo_sim import LearnedModel_BalooSim

data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/validation_inputdata.npy"
# data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/good_data/noisy_press_smooth_joints/"
# data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/good_data/original_data/validation_inputdata.npy"
# data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/good_data/test_data/validation_inputdata.npy"



pred_tester = PredictionValidation(LearnedModel_BalooSim, 3500, BalooSim(XML_PATH="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_baloo.xml"), data_path=data_path, states_of_interest=list(range(18, 24)), data_start_location=5000)
pred_tester.run_trials(
                    "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/run_logs/",
                  "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/test_results/",
                  "noisy_tests")

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# [12, 13, 14, 15, 16, 17]
# [18, 19, 20, 21, 22, 23]