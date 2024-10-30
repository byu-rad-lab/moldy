from pytorch_lightning import seed_everything
import os, sys

from moldy.model.train import train_no_tune
from moldy.case_studies.grub_sim.lightning_module_grub_sim import GrubSimLightningModule

grub_config = {
    "run_name": "data_test",
    "nn_arch": "simple_fnn",
    "b_size": 1024,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "smooth_L1",
    "opt": "adam", 
    "n_hlay": 0,
    "hdim": 2048,
    "act_fn": "relu",
    "lr": 9.9e-4,
    "initialization_scheme": "xavier_uniform",
    "lr_schedule": "reduce_on_plateau",  # "constant", "linear", "cyclic", "reduce_on_plateau",
    "n_inputs": 12,  # [p, qd, q, u] at t
    "n_outputs": 8,  # [p, qd, q] at t+1
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
    "max_epochs": 400,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",

    "vary_params_percent": 0.0,

}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    import numpy as np
    data_amounts = [2000000, 1000000, 500000, 150000]

    data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/grub_sim/data/good_data/"

    for data_amount in data_amounts:
        train_inputdata = np.load(f"{data_path}/train_inputdata.npy")
        train_outputdata = np.load(f"{data_path}/train_outputdata.npy")
        validation_inputdata = np.load(f"{data_path}/validation_inputdata.npy")
        validation_outputdata = np.load(f"{data_path}/validation_outputdata.npy")
        output_max = np.load(f"{data_path}/output_max.npy")

        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/train_inputdata.npy", train_inputdata[:int(0.8*data_amount)])
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/train_outputdata.npy", train_outputdata[:int(0.8*data_amount)])
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/validation_inputdata.npy", validation_inputdata[:int(0.2*data_amount)])
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/validation_outputdata.npy", validation_outputdata[:int(0.2*data_amount)])
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/output_max.npy", output_max)

        grub_config["run_name"] = f"grub_2048net_{data_amount}"
        grub_config["already_trained"] = False
        train_no_tune(grub_config, GrubSimLightningModule)
            