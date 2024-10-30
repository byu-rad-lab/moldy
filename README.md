# moldy

This repository is released with the following paper submitted to RoboSoft 2024: MoLDy: Open-source Library for Data-based Modeling and Nonlinear Model Predictive Control of Soft Robots. For more information see this [video](https://bit.ly/moldy_video).

## Setup
To install dependencies run the following command outside of the package directory

```
pip install -r moldy/requirements.txt
```

## Examples
Each of the case studies in moldy/case_studies/ has python scripts that can be run to optimize hyperparameters and train models. There is also an validation_example.ipynb that shows how to use NEMPC and Prediction validation to evaluate different models generated during optimization and training.


## Setting up a new case study

To use MoLDy with a novel case study you will have to implement the following files (see Inverted Pendulum and Bellows Grub Sim for simulation systems, and Bellows Grub Hardware for hardware systems). Most of these files require writing less than 5 lines of code which is easily outlined in the current case studies.
```
|-- case_studies
|   |-- new_dynamic_system
|   |   |-- data/
|   |   |-- run_logs/
|   |   |   |-- best_models/          -> empty directory
|   |   |   |-- run_logs/             -> empty directory
|   |   |   |-- test_results/         -> empty directory
|   |   +-- model_<system_name>.py    -> contains analytical model of the system
|   |   +-- optimize_<system_name>.py -> code to optimize hyperparameters for a learned dynamic model
|   |   +-- train_<system_name>.py    -> code to train a learned dynamic model given parameters
|   |   +-- lightning_module_<system_name>.py -> code used to train/manage the learned model
|   |   +-- learnedModel_<system_name>.py     -> code to model dynamics using dynamics
|   |   +-- nempc_<system_name>.py            -> code to run nempc with the system (optional for validation)
```

The following hyperparameters can be optimized:
* n_hlay - hidden layers, may mean something different based on the nn_arch
* hdim - hidden nodes in each hidden layer
* lr - learning rate
* act_fn - activation function
* loss_fn - loss function
* opt - optimizer
* initialization_scheme
* lr_schedule - learning rate schedule