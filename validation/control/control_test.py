import numpy as np
import os
from tqdm.auto import tqdm 
import csv
import sys
from datetime import datetime
import matplotlib.pyplot as plt

from moldy.model.Model import Model
from moldy.validation.control.nempc.nempc_base import nempc_run, nempc_conditions_setup

class ControlTest():
    def __init__(self, 
        ground_truth: Model, 
        sim_seconds:float=2.0, 
        **kwargs
        ):
        """
        Sets up control test for a given system
        :param ground_truth: Model, ground truth model to compare against
        :param sim_seconds: float, number of seconds to run control for
        """
        self.sim_seconds = sim_seconds

        self.xgoal_path = kwargs.get("xgoal_path", None)
        self.states_of_interest = kwargs.get("states_of_interest", None)
        self.num_step_commands = kwargs.get("num_step_commands", 1)

        self.x0, self.xgoal_original = nempc_conditions_setup(ground_truth, n=self.num_step_commands)

        if self.xgoal_path is not None:
            self.xgoal_original = np.load(self.xgoal_path)
        self.results = {}

    def run_trials(self, save_path:str, logs_path:str, run_analytical:bool, nempc_setup:callable, nempc_params:dict, test_name:str="", ctrl_dt:float=0.01) -> None:
        """
        Runs control test for analytical and learned models
        :param save_path: str, path to save control data to
        :param logs_path: str, path to directory containing learned models
        :param run_analytical: bool, whether to run control for analytical model
        :param nempc_setup: callable, function to setup NEMPC controller
        :param nempc_params: dict, dictionary containing NEMPC parameters
        :param test_name: str, name of test
        :param ctrl_dt: float, time step for control
        :return: None
        """
        save_dir = save_path + datetime.now().strftime("Control_%Y-%m-%d_%H-%M-%S") + test_name
        try:
            os.makedirs(save_dir)
        except Exception as e:
            print(f"Could not make directory {save_dir} due to {e}")
        
        sys.stdout = open(save_dir+"/terminal_output.txt", 'w')

        if self.num_step_commands > 1:
            self.x0 = self.x0[0].reshape(1, -1)
            self.xgoal = np.repeat(self.xgoal_original, self.sim_seconds/ctrl_dt/self.num_step_commands, axis=0)

        if run_analytical:
            analytical_iae = self.run_analytical(save_dir)
        else:
            analytical_iae = None
        learned_results = self.run_learned_models(logs_path, save_dir, nempc_setup)
        self.save_control_data(save_dir+"/", analytical_iae, learned_results)

        print(f'NEMPC Parameters: \n \tnumSimulations: {nempc_params["numSims"]}, \n\tnumParents: {nempc_params["numParents"]}, \n\tnumStrangers: {nempc_params["numStrangers"]}\n \
            \tHorizon: {nempc_params["horizon"]}, \n\tMutation Probability: {nempc_params["mutation_probability"]}, \n\tCrossover Method: {nempc_params["crossover_method"]} \
                        \n\tSelection Method: {nempc_params["selection_method"]}, \n\tTournament Size: {nempc_params["tournament_size"]}, \
                        \n\tQ: {nempc_params["Q"]}, \n\tR: {nempc_params["R"]} \
                \n\n\n\n')
        
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def run_analytical(self, save_dir:str, nempc_setup:callable) -> None:
        """
        Runs control for analytical model
        :param save_dir: str, path to save control data to
        :param nempc_setup: callable, function to setup NEMPC controller
        :return: None
        """
        print(f'Running Control for Analytical Trial')
        controller, system, ground_truth, x0, _ = nempc_setup(x0=self.x0)
        iae_score, x_history, u_history, t_history = nempc_run(controller, system, ground_truth,
                                                    x0, self.xgoal, 
                                                    sim_length=self.sim_seconds)
        
        print(f'Analytical average IAE={iae_score}')
        self.plot_control_performance("analytical", x_history, self.xgoal, system.numStates, save_dir)
        self.results["analytical"] = {"iae": iae_score, "x_history": x_history, "u_history": u_history, "t_history": t_history}
        return iae_score

    def run_learned_models(self, INPUT_DIR:str, save_dir:str, nempc_setup:callable) -> dict:
        """
        Runs control for all learned models in INPUT_DIR
        :param INPUT_DIR: str, path to directory containing learned models
        :param save_dir: str, path to save control data to
        :param nempc_setup: callable, function to setup NEMPC controller
        :return: dict, dictionary containing trial names, models, iae_metrics
        """
        iae_metrics = []
        trial_names = []
        models = []

        for trial in os.listdir(INPUT_DIR):
            if os.path.isdir(INPUT_DIR + trial):
                trial_dir = INPUT_DIR + trial
                try:
                    print(f'\n\nRunning Control for Trial: {trial}')
                    iae_score, system, x_history, u_history = self.run_single_learned_model(trial_dir, nempc_setup)
                    iae_metrics.append(iae_score)
                    trial_names.append(trial)
                    models.append(system)
                    self.plot_control_performance(trial, x_history, self.xgoal, u_history, system.numStates, save_dir)
                    self.results[trial] = {"iae": iae_score, "x_history": x_history}

                except Exception as e:
                    print(f'For trial {trial} got {e}')

        print(f'Total Learned Model Trials ran: {len(iae_metrics)}')

        if self.states_of_interest is not None:
            sorted_iaes = sorted(zip(trial_names, iae_metrics), key=lambda x: np.sum(np.abs(x[1][0, self.states_of_interest])))
        else:
            sorted_iaes = sorted(zip(trial_names, iae_metrics), key=lambda x: np.sum(np.abs(x[1])))

        print("\n\nSorted IAEs")
        for trial, iae in sorted_iaes:
            if self.states_of_interest is not None:
                print(f"Trial {trial}: {np.sum(np.abs(iae[0, self.states_of_interest]))}")
            else:
                print(f"Trial {trial}: {np.sum(np.abs(iae))}")

        return {"trial_names": trial_names, "models": models, "iae_metrics": iae_metrics}

    def run_single_learned_model(self, trial_dir:str, nempc_setup:callable) -> None:
        """
        Runs control for a single learned model
        :param trial_dir: str, path to directory containing learned model
        :param nempc_setup: callable, function to setup NEMPC controller
        :return: None
        """
        controller, system, ground_truth, x0, _ = nempc_setup(trial_dir=trial_dir,
                                                                        x0=self.x0
                                                                        )
        iae_score, x_history, u_history, t_history = nempc_run(controller, system, ground_truth,
                                                                x0, self.xgoal, 
                                                                sim_length=self.sim_seconds)
        print(f"Average Solve Time: {np.mean(t_history)} s")
        print(f"Final Position: {x_history[-1]}")
        print(f'Average IAE={iae_score}')

        return iae_score, system, x_history, u_history
    
    def save_control_data(self, save_path:str, analytical_iae:np.ndarray, learned_results:dict) -> None:
        """
        Saves control data to csv file
        :param save_path: str, path to save csv file to
        :param analytical_iae: np.ndarray, analytical iae scores
        :param learned_results: dict, dictionary containing trial names, models, iae_metrics
        :return: None
        """
        row_list = []
        models = learned_results["models"]
        iae_metrics = learned_results["iae_metrics"]
        trial_names = learned_results["trial_names"]

        row_list.append(["trial_name", "model_loss", "average_iae_score"])
        if analytical_iae is not None:
            row_list.append(["analytical", "N/A", analytical_iae])
        for model, trial, iae in zip(models, trial_names, iae_metrics):
            print(f"{trial}, {model.get_model_loss()}, {iae}")
            row_list.append([trial, model.get_model_loss(), iae])

        with open(save_path+"control_test_data.csv", 'w', newline='') as file:        
            writer = csv.writer(file)
            writer.writerows(row_list)

    def plot_control_performance(self, trial_name:str, x_history:np.ndarray, goal:np.ndarray, u_history:np.ndarray, numStates:int, save_path:str) -> None:
        """
        Plots the control performance of a learned model
        :param trial_name: str, name of trial to plot
        :param x_history: np.ndarray, state history of size (trajectory_length, num_states)
        :param goal: np.ndarray, goal state of size (trajectory_length, num_states)
        :param u_history: np.ndarray, control history of size (trajectory_length, numInputs)
        :param numStates: int, number of states in the system
        :param save_path: str, path to save plot to
        :return: None
        """
        if self.states_of_interest is None:
            states = list(range(numStates))
        else:
            states = self.states_of_interest
        fig, axs = plt.subplots(len(states), 1, figsize=(10, 5*len(states)), sharex=True)
        t = np.linspace(0, self.sim_seconds, x_history.shape[0])
        for i, state in enumerate(states):
            axs[i].plot(t, x_history[:,state])
            if goal.shape[0] == x_history.shape[0]:
                axs[i].plot(t, goal[:,state], 'r--')
            else:
                axs[i].plot(t, goal[0,state]*np.ones(x_history.shape[0]), 'r--')
            axs[i].set_ylim([-np.pi/2, np.pi/2])
            axs[i].set_title(f"State {state} for {trial_name}")
            axs[i].set_ylabel("State Value")
            axs[i].legend(["Learned", "Goal"])

        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(save_path+f"/control_performance_{trial_name}.png")
        plt.close()

        fig, axs = plt.subplots(u_history.shape[1], 1, figsize=(10, 5*u_history.shape[1]), sharex=True)
        for i in range(u_history.shape[1]):
            axs[i].plot(t, u_history[:,i])
            axs[i].set_title(f"Control {i} for {trial_name}")
            axs[i].set_ylabel("Control Value")
            axs[i].legend([f"Control {i}"])

        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(save_path+f"/control_performance_controls_{trial_name}.png")
        plt.close()