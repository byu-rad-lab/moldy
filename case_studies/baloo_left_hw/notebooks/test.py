from moldy.validation.prediction.prediction_validation import PredictionValidation
from moldy.case_studies.baloo_left_hw.learnedModel_baloo_hw import LearnedModel_BalooHW

# from moldy.case_studies.baloo_left_hw.learnedModel_baloo_hw_pressures import LearnedModel_BalooHW

data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/validation_inputdata.npy"
# data_path ="/home/daniel/Documents/data/daniel_baloo_data_collection/STEPCMDS/parsed_csvs/smooth_all_inputs_normalized_filtered.npy"
# data_path ="/home/daniel/Documents/data/daniel_baloo_data_collection/STEPCMDS/parsed_csvs/smooth_all_inputs_normalized.npy"

ground_truth = LearnedModel_BalooHW(trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/results/best_models/daniels_smooth_L1_weighted_mult1_10_mult2_5")

tester = PredictionValidation(LearnedModel_BalooHW, 3000, ground_truth, states_of_interest=list(range(18, 24)), data_path=data_path, data_start_location=10000)
tester.run_trials(
                "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/results/run_logs/",
                  # "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/",
                  "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/results/test_results/",
                  "xfer_to_smooth_prev_models")

# [18, 19, 20, 21, 22, 23]