from pytorch_lightning import seed_everything
from ray import tune
import os, sys

from moldy.model.optimize import optimize_model
from moldy.case_studies.inverted_pendulum.lightning_module_ip import InvertedPendulumLightningModule

ip_config = {
    "project_name": "moldy_ip_sim",
    "run_name": "test",
    "nn_arch": "simple_fnn",
    "b_size": 512,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "mse",
    # ============================= Parameters to Optimize ===============================
    "n_hlay": tune.choice(list(i for i in range(3))),
    "hdim": tune.choice(list(2**i for i in range(2, 11))),
    "lr": tune.loguniform(1e-5, 1e-3),
    "act_fn": tune.choice(["relu", "tanh", "sigmoid", "leaky_relu"]),
    "opt": tune.choice(["adam", "sgd"]),
    "initialization_scheme": tune.choice(["uniform", "normal", "xavier_uniform", "xavier_normal"]),
    "lr_schedule": tune.choice(["constant", "linear", "cyclic", "reduce_on_plateau"]),
    # ============================= Data Parameters ======================================
    "n_inputs": 3,  # [theta_dot, theta, tau] at t
    "n_outputs": 2,  # [theta_dot, theta] at t+1
    
    "data_generation_params": {
        "type": "step",
        "dataset_size": 50000,
        "generate_new_data": False,
        "normalization_method": "mean_std",  # "min_max", "standardized", "mean_std", "standard", "none"
        "learn_mode": "delta_x",  # "x", "delta_x"
        "dt": 0.01,
    },
    # ============================ Optimization/Training Parameters =========================
    "num_workers": 6,
    "cpu_num": 8,
    "gpu_num": 1.0,
    "max_epochs": 150,
    "num_samples": 150,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",
}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    optimize_model(
        ip_config,
        InvertedPendulumLightningModule,
        trial_dir=None,
    )
