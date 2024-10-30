import numpy as np
import os, sys
import torch
from typing import Tuple
from datetime import datetime
np.set_printoptions(precision=3, suppress=True)

from moldy.model.LearnedModel import LearnedModel
from moldy.model.Model import Model


class PredictionValidation():
    def __init__(self, LearnedModelObject:LearnedModel, trajectory_length:int, ground_truth:Model, data_path:str=None, states_of_interest:np.ndarray=None, data_start_location:int=0):
        """
        Sets up Prediction validation for a given system
        :param LearnedModelObject: LearnedModel, LearnedModel object to run prediction validation on
        :param trajectory_length: int, length of trajectory to run prediction validation on
        :param ground_truth: Model, ground truth model to compare against
        :param data_path: str, path to directory containing numpy arrays to run prediction validation on
            - For simulation systems this is set to None and the ground_truth will be run for trajectory_length steps
        :param states_of_interest: np.ndarray, states to calculate IAE for
        
        If data_path is not none, data from that path will be loaded as the ground truth data to compare
        learned model performance against. Otherwise, ground_truth should be provided to run the analytical model
        """
        self.LearnedModelObject = LearnedModelObject
        self.trajectory_length = trajectory_length
        self.ground_truth = ground_truth
        self.data_path = data_path
        self.states_of_interest = states_of_interest
        self.data_start_location = data_start_location
        self.results = {}
        
    def get_ground_truth_data_from_file(self, learnedModel) -> np.ndarray:
        """
        Gets ground truth data from saved file
        :return: np.ndarray, ground truth data of size (trajectory_length, num_states + num_inputs)
        """
        input_data = np.load(self.data_path)
        if learnedModel.normalization_method == "max":
            input_data = input_data * torch.hstack([learnedModel.xMax, learnedModel.uMax]).cpu().numpy()
        elif "mean_std" in learnedModel.normalization_method:
            input_data[:, :self.ground_truth.numStates] = (input_data[:, :self.ground_truth.numStates] * learnedModel.state_std.cpu().numpy()) + learnedModel.state_mean.cpu().numpy()
            input_data[:, self.ground_truth.numStates:] = (input_data[:, self.ground_truth.numStates:] * learnedModel.input_std.cpu().numpy()) + learnedModel.input_mean.cpu().numpy()
        elif "min_max" in learnedModel.normalization_method:
            input_data[:, :learnedModel.numStates] = (input_data[:, :learnedModel.numStates] * learnedModel.state_max.cpu().numpy()) * learnedModel.state_std.cpu().numpy() + learnedModel.state_mean.cpu().numpy()
            input_data[:, learnedModel.numStates:] = (input_data[:, learnedModel.numStates:] * learnedModel.input_max.cpu().numpy()) * learnedModel.input_std.cpu().numpy() + learnedModel.input_mean.cpu().numpy()
    
        self.u0 = input_data[self.data_start_location, self.ground_truth.numStates:]

        return input_data[self.data_start_location:self.data_start_location+self.trajectory_length, :]
    
    def get_ground_truth_data_from_analytical(self, dt:float=0.01) -> np.ndarray:
        """
        Gets ground truth data from analytical model to compare learned model against
        :param dt: float, time step to run ground truth data at
        :return: np.ndarray, ground truth data of size (trajectory_length, num_states + num_inputs)
        """
        analytical_inputs = np.zeros((self.trajectory_length, self.ground_truth.numStates+self.ground_truth.numInputs))
        self.x0 = self.ground_truth.generate_random_state(initial_conditions=True)
        self.u0 = self.ground_truth.generate_random_command()

        x_analytical = self.x0
        u = self.u0

        analytical_inputs[0] = np.hstack([x_analytical, u]).flatten()
        for i in range(self.trajectory_length):
            if i % 500 == 0:
                u = self.ground_truth.generate_random_command()
            x_analytical = self.ground_truth.forward_simulate_dt(x_analytical, u, dt)
            analytical_inputs[i] = np.hstack([x_analytical, u]).flatten()

        return analytical_inputs
    
    def run_trials(self, logs_path:str, save_path:str, trial_name:str="", dt:float=0.01) -> None:
        """
        Runs prediction tests on all learned models in logs_path and saves results to save_path
        :param save_path: str, path to directory to save results to
        :param logs_path: str, path to directory containing learned models
        :param trial_name: str, name of trial to save results under
        :param dt: float, time step to run learned models at
        :return: None
        """
        trial_name = "Prediction" if trial_name == "" else trial_name
        save_dir = save_path + datetime.now().strftime(f"P%Y-%m-%d_%H-%M_%S_{trial_name}")
        try:
            os.makedirs(save_dir)
        except Exception as e:
            print(f"Could not make directory {save_dir} due to {e}")

        sys.stdout = open(save_dir+"/terminal_output.txt", 'w')
        
        self.run_learned_models(logs_path, save_dir, dt)

        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def run_learned_models(self, logs_path:str, save_path:str, dt:float=0.01) -> dict:
        """
        Runs multiple learned models saved in logs_path to compare prediction across models
        :param logs_path: str, path to directory containing learned models
        :param dt: float, time step to run learned models at
        :param save_plots: bool, whether or not to save plots of learned model performance

        :return: dict, dictionary containing trial names, models, iae_metrics, learned_inputs, analytical_results
        """
        iae_metrics = []
        trial_names = []
        models = []
        learned_predictions_all = []

        for trial_name in os.listdir(logs_path):
            if os.path.isdir(logs_path + trial_name):
                trial_dir = logs_path + trial_name
                try:
                    learned_model, iae, learned_predictions, ground_truth_data = self.run_prediction_single_model(trial_dir, dt)
                    print(f"\n\nTrial: {trial_name}, \n\tIAE: {iae}")
                    trial_names.append(trial_name)
                    models.append(learned_model)
                    iae_metrics.append(iae)
                    learned_predictions_all.append(learned_predictions)
                    self.plot_prediction_performance(trial_name, learned_predictions, save_path, ground_truth_data)
                except Exception as e:
                    print(f'For trial {trial_name} got exception: {e}')
                    continue

        print(f'Total Trials ran: {len(iae_metrics)}')

        # print trials in order of lowest iae
        if self.states_of_interest is not None:
            sorted_iaes = sorted(zip(trial_names, iae_metrics), key=lambda x: np.sum(np.abs(x[1][0, self.states_of_interest])))
        else:
            sorted_iaes = sorted(zip(trial_names, iae_metrics), key=lambda x: np.sum(np.abs(x[1])))

        print("\n\nSorted IAEs")
        for trial, iae in sorted_iaes:
            if self.states_of_interest is not None:
                print(f"Trial {trial}: IAE: {np.sum(np.abs(iae[0, self.states_of_interest]))}")
                print(f"\tTotal iae: {np.sum(np.abs(iae))}, Pressure IAE: {np.sum(np.abs(iae[0, :4]))}, Velocity IAE: {np.sum(np.abs(iae[0, 4:6]))}, Position IAE: {np.sum(np.abs(iae[0, 6:8]))}")
                print(f"\tPressure/Velocity/Angle: {(np.sum(iae[0, :4]))/self.trajectory_length/4} kPa  & {np.sum(iae[0, 4:6])/self.trajectory_length/2} rad/s  & {np.sum(iae[0, 6:8])/self.trajectory_length/2} rad")
                try:
                    print(f"\tPressure/Velocity/Angle Baloo: {(np.sum(iae[0, :12]))/self.trajectory_length/12} kPa  & {np.sum(iae[0, 12:18])/self.trajectory_length/6} rad/s  & {np.sum(iae[0, 18:24])/self.trajectory_length/6} rad")
                except:
                    pass
            else:
                print(f"Trial {trial}: {np.sum(np.abs(iae))}")
    
    def run_prediction_single_model(self, trial_dir:str, dt:float) -> Tuple[LearnedModel, np.ndarray, np.ndarray]:
        """"
        Runs a single learned model to compare prediction against ground truth
        :param trial_dir: str, path to directory containing learned model
        :param dt: float, time step to run learned model at
        :return: LearnedModel, np.ndarray, np.ndarray, learned model, iae metric, learned predictions
        """
        learned_model = self.LearnedModelObject(trial_dir=trial_dir)

        ground_truth_data = self.get_ground_truth_data_from_file(learned_model)

        learned_predictions = np.zeros((self.trajectory_length,learned_model.numStates))
        learned_predictions[0] = ground_truth_data[0,:learned_model.numStates].flatten()
        x_learned = torch.from_numpy(ground_truth_data[0,:learned_model.numStates].reshape(1, -1)).float().cuda()

        for i in range(1,self.trajectory_length):
            x_learned = learned_model.forward_simulate_dt(
                                            x_learned, 
                                            torch.from_numpy(ground_truth_data[i-1,-learned_model.numInputs:].reshape(1, -1)).float().cuda())
            learned_predictions[i] = x_learned.cpu().detach().numpy()
        iae = np.sum(np.abs(ground_truth_data[:,:learned_model.numStates] - learned_predictions), axis=0).reshape(1, -1)
        self.results[trial_dir] = {"iae": iae, "learned_predictions": learned_predictions}

        self.results["ground_truth"] = ground_truth_data

        return learned_model, iae, learned_predictions, ground_truth_data

    def plot_prediction_performance(self, trial_name:str, learned_predictions:np.ndarray, save_path:str, ground_truth_data:np.ndarray) -> None:
        """
        Plots the prediction performance of a learned model
        :param trial_name: str, name of trial to plot
        :param learned_predictions: np.ndarray, learned predictions of size (trajectory_length, num_states)
        :param save_path: str, path to save plot to

        :return: None
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,3*self.ground_truth.numStates))
        if save_path[-1] != '/':
            save_path += '/'
        for i in range(self.ground_truth.numStates):
            plt.subplot(self.ground_truth.numStates,1,i+1)
            if i < self.ground_truth.numInputs:
                plt.plot(ground_truth_data[:,i+self.ground_truth.numStates], label="Command")
            plt.plot(ground_truth_data[:,i], label="Ground Truth")
            plt.plot(learned_predictions[:,i], linestyle="--",label="Learned Model")
            plt.ylabel(f"State {i+1}")
            plt.tight_layout()
            plt.grid()
            plt.legend()
        plt.xlabel("Time Step")
        plt.savefig(save_path + trial_name + '.png')
        plt.close()