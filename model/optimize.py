from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import os
import csv, json
import numpy as np
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass
from moldy.model.train import train_tune

def optimize_model(config: dict, lighting_module_given: LightningModuleBaseClass, trial_dir: str=None) -> None:
    """
    Optimize hyperparameters for a given system
    :param config: dictionary containing the configuration for the model
    :param lighting_module_given: LightningModuleBaseClass object
    :param trial_dir: directory containing the trial to restore if applicable
    :return: None
    """
    train_fn_with_parameters = tune.with_parameters(
        train_tune, lighting_module_given=lighting_module_given
    )
    resources_per_trial = {"cpu": config.get("cpu_num", 8.0), "gpu": config.get("gpu_num", 1.0)}

    if trial_dir is None:
        scheduler = ASHAScheduler(
            max_t=config.get("max_epochs", 150), grace_period=1, reduction_factor=3
        )
        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, resources_per_trial),
            param_space=config,
            tune_config=tune.TuneConfig(
                search_alg=OptunaSearch(),
                scheduler=scheduler,
                metric=config.get("metric", "val_loss"),
                mode=config.get("mode", "min"),
                num_samples=config.get("num_samples", 100),
            ),
            run_config=air.RunConfig(
                name=config["run_name"],
                progress_reporter=CLIReporter(metric_columns=config["metric"]),
                storage_path=config["path"] + "results/run_logs/",
            ),
        )
    else:
        tuner = tune.Tuner.restore(
            trial_dir,
            tune.with_resources(train_fn_with_parameters, resources_per_trial),
        )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

def analyze_optimization(run_path:str) -> None:
    """
    Analyze the results of an optimization run
    :param run_path: path to the run logs
    :return: None
    
    Saves a csv file for each trial directory containing the trial name, min loss, min loss epoch, and hyperparameters
    """

    results = {"trial_names": [], "min_losses": [], "min_loss_index": [], "hyperparameters": []}
    all_logs = [trial for trial in os.listdir(run_path) if trial.startswith("train")]
    for run in all_logs:
        try:
            single_result = load_optimized_model(run_path+"/"+run)
            results["trial_names"].append(single_result["trial_name"])
            results["min_losses"].append(single_result["min_loss"])
            results["min_loss_index"].append(single_result["min_loss_index"])
            results["hyperparameters"].append(single_result["config"])
        except Exception as e:
            print(f"Exception loading run due to {e}")

    zipped_results = list(zip(results["min_losses"], results["min_loss_index"], results["trial_names"], results["hyperparameters"]))
    row_list = []
    sorted_hyperparam_names = dict(sorted(results["hyperparameters"][0].items()))
    row_list.append(["trial_name", "min_loss", "min_loss_epoch"] + list(sorted_hyperparam_names.keys()))
    zipped_results.sort(key=lambda x: x[0])

    for trial in zipped_results:
        sorted_hyperparams = dict(sorted(trial[3].items()))
        row_list.append([trial[2], trial[0], trial[1]] + list(sorted_hyperparams.values()))

    with open(run_path+f"/optimization_analysis.csv", 'w', newline='') as file:        
        writer = csv.writer(file)
        writer.writerows(row_list)
    print(f"Average Min Loss Epoch: {np.mean(results['min_loss_index'])}")
    print(f"{len(row_list)-1} trials saved to {run_path}/{run_path.split('/')[-1]}_analysis.csv")

def load_optimized_model(trial_dir:str) -> dict:
    """
    Load the best model from a trial directory
    :param trial_dir: path to the trial directory
    :return: dictionary containing the trial name, hyperparameters, min loss, min loss epoch, and checkpoint path
    """
    trial_name = trial_dir.split("/")[-1]
    checkpoint_path = trial_dir + "/lightning_logs/version_0/checkpoints/lowest_loss.ckpt"

    with open(trial_dir+"/params.json", "r") as f:
        config = json.load(f)
    metrics_path = trial_dir + "/progress.csv"
    
    with open(metrics_path, "r") as f:
        resultList = []
        csv_reader = csv.reader(f, delimiter=",")
        loss_idx = 0
        for row in csv_reader:
            if not row[0][0].isalpha():
                loss = float(row[loss_idx])
                resultList.append(loss)
            else:
                loss_idx = row.index(next(item for item in row if 'loss' in item))

    minLoss = np.nanmin(np.array(resultList))
    minLossIndex = np.nanargmin(np.array(resultList))
    
    return {"trial_name": trial_name, 
            "config": config, 
            "min_loss": minLoss, 
            "min_loss_index": minLossIndex, 
            "checkpoint_path": checkpoint_path}

if __name__=="__main__":
    run_path = "/home/daniel/catkin_ws/src/moldy/case_studies/arm_sim/results/run_logs/run4_500K"

    analyze_optimization(run_path)