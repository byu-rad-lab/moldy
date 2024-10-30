from pytorch_lightning import seed_everything
from ray import tune
import os, sys

from moldy.model.optimize import optimize_model
from moldy.case_studies.baloo_left_hw.lightning_module_baloo_hw import BalooHWLightningModule

baloo_sim_config = {
    # "run_name": "run1_1M",
    "nn_arch": "simple_fnn",
    "b_size": 512,
    "metric": "val_loss",
    "mode": "min",
    # "loss_fn": "mae",
    # ============================= Parameters to Optimize ===============================
    "n_hlay": tune.choice(list(i for i in range(8))),
    "hdim": tune.choice(list(2**i for i in range(3, 11))),
    "lr": tune.loguniform(1e-6, 1e-3),
    "act_fn": tune.choice(["relu", "tanh", "sigmoid", "leaky_relu"]),
    "opt": tune.choice(["adam", "sgd"]),  # 'ada', 'lbfgs', 'rmsprop']),
    "lr_schedule": "reduce_on_plateau", # tune.choice(["constant", "linear", "cyclic", "reduce_on_plateau"]),
    "initialization_scheme": "xavier_uniform", # tune.choice(["uniform", "normal", "xavier_uniform", "xavier_normal"]),
    # ============================= Data Parameters ======================================
    "n_inputs": 36,  # [p, qd, q, u] at t
    "n_outputs": 24,  # [p, qd, q] at t+1

    "data_generation_params": {
        "type": "step",
        "dataset_size": 0,
        "generate_new_data": False,
        "normalization_method": "mean_std",  # "min_max", "standardized", "mean_std", "standard", "none"
        "learn_mode": "delta_x",  # "x", "delta_x"
        "dt": 0.02,
    },

    # ============================ Optimization/Training Parameters =========================
    "num_workers": 8,
    "cpu_num": 8,
    "gpu_num": 1.0,
    "max_epochs": 200,
    "num_samples": 100,
    "path": f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/",
}

if __name__ == "__main__":
    seed_everything(42, workers=True)

    loss_options = ["mae"]

    for loss in loss_options:
        baloo_sim_config["loss_fn"] = loss
        baloo_sim_config["run_name"] = f"{loss}_optimization"

        trial_dir = None
        if os.path.exists(f"/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/run_logs/{loss}_optimization"):
            trial_dir = f"/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/run_logs/{loss}_optimization"

        optimize_model(
            baloo_sim_config,
            BalooHWLightningModule,
            trial_dir=trial_dir,
        )