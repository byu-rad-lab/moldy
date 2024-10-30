from pytorch_lightning import seed_everything
from ray import tune
import os, sys

from moldy.model.optimize import optimize_model
from moldy.case_studies.grub_hw.lightning_module_grub_hw import GrubHWLightningModule

grub_hw_config = {
    "project_name": "moldy_grub_hw",
    "run_name": "test",
    "nn_arch": "simple_fnn",
    "b_size": 1024,
    "metric": "val_loss",
    "mode": "min",
    "loss_fn": "mse",
    # ============================= Parameters to Optimize ===============================
    "n_hlay": tune.choice(list(i for i in range(4))),
    "hdim": tune.choice(list(2**i for i in range(4, 11))),
    "lr": tune.loguniform(1e-5, 1e-3),
    "act_fn": tune.choice(["relu", "tanh", "sigmoid", "leaky_relu"]),
    "opt": tune.choice(["adam", "sgd"]),  # 'ada', 'lbfgs', 'rmsprop']),
    "initialization_scheme": tune.choice(["uniform", "normal", "xavier_uniform", "xavier_normal"]),
    "lr_schedule": tune.choice(["constant", "linear", "cyclic", "reduce_on_plateau"]),
    # ============================= Data Parameters ======================================
    "n_inputs": 12,  # [p, qd, q, u] at t
    "n_outputs": 8,  # [p, qd, q] at t+1

    "data_generation_params": {
        "type": "step",
        "dataset_size": 0,
        "generate_new_data": False,
        "normalization_method": "mean_std",  # "min_max", "standardized", "mean_std", "standard", "none"
        "learn_mode": "delta_x",  # "x", "delta_x"
        "dt": 0.02,
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
            grub_hw_config,
            GrubHWLightningModule,
            trial_dir=None
        )