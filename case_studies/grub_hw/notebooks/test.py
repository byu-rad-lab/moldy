from moldy.validation.prediction.prediction_validation import PredictionValidation
from moldy.case_studies.grub_hw.learnedModel_grub_hw import LearnedModel_GrubHW

data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/good_data/max_normalization/raw_data/validation_inputdata.npy"
# data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/good_data/prev_data/smooth/validation_inputdata.npy"
# data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/notebooks/data.npy"

ground_truth = LearnedModel_GrubHW("/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/results/best_models/Sim")

pred_tester = PredictionValidation(LearnedModel_GrubHW, 5000, ground_truth, data_path=data_path, states_of_interest=[6, 7], data_start_location=5000)
pred_tester.run_trials(
                "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/results/best_models/",
    # "/home/daniel/catkin_ws/src/moldy/case_studies/grub_sim/results/run_logs/",
                                  "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/results/test_results/",
                                  "PAPER_PREDICTION_SMALLER_VALIDATION_SET")