import numpy as np
import torch

from moldy.model.LearnedModel import LearnedModel
from moldy.case_studies.grub_hw.lightning_module_grub_hw import GrubHWLightningModule
from moldy.case_studies.grub_sim.model_grub_sim import GrubSim

class LearnedModel_GrubHW(LearnedModel, GrubSim):
    def __init__(self, trial_dir:str=None, **kwargs):
        super().__init__(LightningModule=GrubHWLightningModule, trial_dir=trial_dir, **kwargs)
        self.name = "LearnedModel_GrubHW"


if __name__ == "__main__":
    trial_dir = "/home/daniel/catkin_ws/src/moldy/case_studies/grub_hw/results/best_models/version_17"
    model = LearnedModel_GrubHW(trial_dir)
