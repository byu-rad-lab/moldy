import torch
import numpy as np

from moldy.model.LearnedModel import LearnedModel
from moldy.case_studies.baloo_left_hw.lightning_module_baloo_hw import BalooHWLightningModule
from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim

class LearnedModel_BalooHW(LearnedModel, BalooSim):
    def __init__(self, trial_dir:str=None, **kwargs):
        super().__init__(LightningModule=BalooHWLightningModule, trial_dir=trial_dir, **kwargs)
        self.name = "LearnedModel_BalooHW"



if __name__ == "__main__":
    trial_dir = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/results/run_logs/base_0"
    model = LearnedModel_BalooHW(trial_dir)
