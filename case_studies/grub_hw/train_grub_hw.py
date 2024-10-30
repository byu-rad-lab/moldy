from pytorch_lightning import seed_everything
import os, sys

from moldy.model.train import train_no_tune
from moldy.case_studies.grub_hw.lightning_module_grub_hw import GrubHWLightningModule

grub_config = {
    "run_name": "base_hardware_model_250epochs",
    "nn_arch": "simple_fnn",
    "b_size": 1024,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "smooth_L1",
    "opt": "adam", 
    "n_hlay": 0,
    "hdim": 2048,
    "act_fn": "relu",
    "lr": 1.0e-5,
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
    "max_epochs": 200,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",
}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    import numpy as np
    data_amounts = [50000, 100000, 200000, 360000]


    # fine tuning training
    epochs = [25, 50, 100, 200]
    lrs = [1.0e-6, 5.0e-6, 1.0e-5, 5.0e-5]

    # Base model training
    # epochs = [100, 200, 300]
    # lrs = [1.0e-4, 5.0e-4]


    base_model_path = "/home/daniel/Documents/data/xfer_learning/sim_to_hw_grub_data/best_models/360K BASE"

    data_path = "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/good_data/max_normalization/smooth/"

    for data_amount in data_amounts:
        train_inputdata = np.load(f"{data_path}/train_inputdata.npy")[:int(0.8*data_amount)]
        train_outputdata = np.load(f"{data_path}/train_outputdata.npy")[:int(0.8*data_amount)]
        validation_inputdata = np.load(f"{data_path}/validation_inputdata.npy")[:int(0.2*data_amount)]
        validation_outputdata = np.load(f"{data_path}/validation_outputdata.npy")[:int(0.2*data_amount)]
        output_max = np.load(f"{data_path}/output_max.npy")

        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/train_inputdata.npy", train_inputdata)
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/train_outputdata.npy", train_outputdata)
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/validation_inputdata.npy", validation_inputdata)
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/validation_outputdata.npy", validation_outputdata)
        np.save(f"/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/data/output_max.npy", output_max)

        for epoch in epochs:
            for lr in lrs:
                grub_config["max_epochs"] = epoch
                grub_config["lr"] = lr

                grub_config["pretrained_model_path"] = base_model_path
                grub_config["run_name"] = f'finetuned_grub_hw_{data_amount}_{grub_config["lr"]}_{grub_config["max_epochs"]}'
                grub_config["already_trained"] = False
                train_no_tune(grub_config,GrubHWLightningModule)