from pytorch_lightning import seed_everything
import os, sys

from moldy.model.train import train_no_tune
from moldy.case_studies.baloo_left_hw.lightning_module_baloo_hw import BalooHWLightningModule

arm_config = {
    "run_name": "HW Model",
    "nn_arch": "simple_fnn",
    "b_size": 2048,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "smooth_L1",
    "opt": "adam", 
    "n_hlay": 0,
    "hdim": 2048,
    "act_fn": "relu",
    "lr": 9.0e-4,
    "initialization_scheme": "xavier_uniform",
    "lr_schedule": "reduce_on_plateau",  # "constant", "linear", "cyclic", "reduce_on_plateau",

    "n_inputs": 36,  # [p, qd, q, u] at t
    "n_outputs": 24,  # [p, qd, q] at t+1
    "num_workers": 8,

    "data_generation_params": {
        "type": "step",
        "dataset_size": 0,
        "generate_new_data": False,
        "normalization_method": "mean_std",  # "min_max", "standardized", "mean_std", "standard", "none"
        "learn_mode": "delta_x",  # "x", "delta_x"
        "dt": 0.01,
    },
    # ============================ Optimization/Training Parameters =========================
    "max_epochs": 200,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",

    # "pretrained_model_path": "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/results/run_logs/tf_test",
    # "pretrained_model_path": "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/daniels_smooth_L1_weighted_mult1_10_mult2_5",
    # "insert_layer": False,
    # "freeze_pretrained": False,

}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    import numpy as np

    # path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/" 
    # lr_options = [1.0e-4, 1.0e-5, 1.0e-6]
    # # max_epoch_options = [10, 25, 50, 100, 200]
    # max_epoch_options = [200]
    # amount_of_data = [1.0]#, 0.5, 0.25, 0.1]

    # # arm_config["pretrained_model_path"] = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/daniels_smooth_L1_weighted_mult1_10_mult2_5"

    # for dataset_size in amount_of_data:
    #     arm_input_train_data = np.load('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/gooddata/smooth/train_inputdata.npy')
    #     arm_output_train_data = np.load('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/gooddata/smooth/train_outputdata.npy')
    #     arm_input_validation_data = np.load('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/gooddata/smooth/validation_inputdata.npy')
    #     arm_output_validation_data = np.load('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/gooddata/smooth/validation_outputdata.npy')

    #     np.save('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/train_inputdata.npy', arm_input_train_data[:int(arm_input_train_data.shape[0]*dataset_size)])
    #     np.save('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/train_outputdata.npy', arm_output_train_data[:int(arm_output_train_data.shape[0]*dataset_size)])
    #     np.save('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/validation_inputdata.npy', arm_input_validation_data[:int(arm_input_validation_data.shape[0]*dataset_size)])
    #     np.save('/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/data/validation_outputdata.npy', arm_output_validation_data[:int(arm_output_validation_data.shape[0]*dataset_size)])

    #     for lr in lr_options:
    #         arm_config["lr"] = lr
    #         for max_epoch in max_epoch_options:
    #             arm_config["max_epochs"] = max_epoch
    #             arm_config["already_trained"] = False
    #             arm_config["run_name"] = f"base_{dataset_size}_{max_epoch}_lr_{lr}"
    train_no_tune(arm_config, BalooHWLightningModule)