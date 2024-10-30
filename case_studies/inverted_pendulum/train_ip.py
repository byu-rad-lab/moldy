from pytorch_lightning import seed_everything
import os, sys


from moldy.model.train import train_no_tune
from moldy.case_studies.inverted_pendulum.lightning_module_ip import InvertedPendulumLightningModule

ip_config = {
    # "run_name": "good_model_with_diff_ip_params2",
    "nn_arch": "simple_fnn",
    # "b_size": 256,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "mse", 
    # ============================= Optimized Parameters ===============================
    "opt": "adam", 
    "n_hlay": 2,
    "hdim": 1024,
    "act_fn": "leaky_relu",
    # "lr": 0.0005,
    "initialization_scheme": "xavier_uniform", #"xavier_uniform",
    "lr_schedule": "reduce_on_plateau", # "reduce_on_plateau"
    # ============================= Data Parameters ======================================
    "n_inputs": 3,  # [theta_dot, theta, tau] at t
    "n_outputs": 2,  # [theta_dot, theta] at t+1
    "dt": 0.01,

    "data_generation_params": {
        "type": "random",
        # "dataset_size": 60000,
        "generate_new_data": False,
        "normalization_method": "mean_std",  # "min_max", "standardized", "mean_std", "standard", "none"
        "learn_mode": "delta_x",  # "x", "delta_x"
        "dt": 0.01,
        },
    # ============================ Optimization/Training Parameters =========================
    "num_workers": 8,
    # "max_epochs": 100,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",
}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    ip_config["pretrained_model_path"] = "/home/daniel/catkin_ws/src/moldy/case_studies/inverted_pendulum/results/run_logs/same_mse_1"

    # create a script that will test each combination of these parameters by training a model
    b_size_options = [256, 512]
    dataset_size_options = [500, 1000, 5000, 10000]
    lr_options = [5.0e-6, 5.0e-7, 5.0e-8]
    max_epoch_options = [10, 20, 50, 100]

    for b_size in b_size_options:
        for dataset_size in dataset_size_options:
            if b_size == 64 and dataset_size == 500:
                continue
            else:
                for lr in lr_options:
                    for max_epoch in max_epoch_options:
                        ip_config["b_size"] = b_size
                        ip_config["data_generation_params"]["dataset_size"] = dataset_size
                        ip_config["lr"] = lr
                        ip_config["max_epochs"] = max_epoch
                        ip_config["run_name"] = f"b_size_{b_size}_dataset_size_{dataset_size}_lr_{lr}_max_epoch_{max_epoch}"
                        train_no_tune(ip_config, InvertedPendulumLightningModule)