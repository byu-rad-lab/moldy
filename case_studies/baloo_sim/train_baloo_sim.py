from pytorch_lightning import seed_everything
import os, sys

from moldy.model.train import train_no_tune
from moldy.case_studies.baloo_sim.lightning_module_baloo_sim import BalooSimLightningModule

arm_config = {
    "run_name": "Noisy Sim",
    "nn_arch": "simple_fnn",
    "b_size": 2048,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "smooth_L1",
    "opt": "adam", 
    "n_hlay": 0,
    "hdim": 2048,
    "act_fn": "relu",
    "lr": 9.9e-04,
    "initialization_scheme": "xavier_uniform",
    "lr_schedule": "reduce_on_plateau",  # "constant", "linear", "cyclic", "reduce_on_plateau",
    "n_inputs": 36,  # [p, qd, q, u] at t
    "n_outputs": 24,  # [p, qd, q] at t+1
    "num_workers": 8,

    "data_generation_params": {
        "type": "step",
        "dataset_size": 0,
        "generate_new_data": False,
        "normalization_method": "max",  # "min_max", "standardized", "mean_std", "standard", "none"
        "learn_mode": "delta_x",  # "x", "delta_x"
        "dt": 0.01,
    },
    # ============================ Optimization/Training Parameters =========================
    "max_epochs": 300,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",

    # "insert_layer": True,
    # "freeze_pretrained": True
    # "weight_decay": 0.0,
}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    # import numpy as np
    # data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/good_data/sys_id_data"

    # data_amounts = [1500000, 1000000, 500000, 250000, 100000]
    # data_amounts = [1000000, 500000, 250000, 100000]

    # BASE MODEL PARAMETERS
    # lrs = [5.0e-4, 1.0e-4, 5.0e-5]
    # epochs = [300, 200]

    # FINE TUNING PARAMETERS
    # epochs = [25, 50, 100, 200]
    # lrs = [1.0e-6, 5.0e-6, 1.0e-5, 5.0e-5, 1.0e-4]


    # for data in data_amounts:
    #     train_inputdata = np.load(f"{data_path}/train_inputdata.npy")
    #     train_outputdata = np.load(f"{data_path}/train_outputdata.npy")
    #     val_inputdata = np.load(f"{data_path}/validation_inputdata.npy")
    #     val_outputdata = np.load(f"{data_path}/validation_outputdata.npy")
    #     output_max = np.load(f"{data_path}/output_max.npy")

    #     np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/train_inputdata.npy", train_inputdata[:int(0.8*data)])
    #     np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/train_outputdata.npy", train_outputdata[:int(0.8*data)])
    #     np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/validation_inputdata.npy", val_inputdata[:int(0.2*data)])
    #     np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/validation_outputdata.npy", val_outputdata[:int(0.2*data)])

    #     for lr in lrs:
    #         for epoch in epochs:
    #             arm_config["max_epochs"] = epoch
    #             arm_config["lr"] = lr
    #             arm_config["run_name"] = f"finetuned_baloo_sim_{data}_{lr}_{epoch}"
    #             arm_config["already_trained"] = False
    #             arm_config["pretrained_model_path"] = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/best_models/SOURCE"

    train_no_tune(arm_config, BalooSimLightningModule)