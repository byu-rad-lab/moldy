import torch
import numpy as np

from moldy.model.LearnedModel import LearnedModel
from moldy.case_studies.baloo_left_hw.lightning_module_baloo_hw import BalooHWLightningModule
from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim

class LearnedModel_BalooHW(LearnedModel, BalooSim):
    def __init__(self, trial_dir:str=None, **kwargs):

        use_gpu = kwargs.get("use_gpu", True)
        super().__init__(LightningModule=BalooHWLightningModule, trial_dir=trial_dir, **kwargs)
        self.name = "LearnedModel_BalooHW"

        self.numStates = 12
        self.numInputs = 12

        self.xMax = torch.tensor([[200.0, 200.0, 200.0, 200.0,
                                   200.0, 200.0, 200.0, 200.0,
                                   200.0, 200.0, 200.0, 200.0,
                                #    np.pi, np.pi, np.pi, np.pi, np.pi, np.pi,
                                #    1.5, 1.5, 1.5, 1.5, 1.5, 1.5
                                   ]]).cuda()
        self.xMin = torch.tensor([[0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0,
                                #   -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi,
                                #   -1.5, -1.5, -1.5, -1.5, -1.5, -1.5
                                      ]]).cuda()
        
        self.uMax = torch.tensor([[200.0, 200.0, 200.0, 200.0,
                                   200.0, 200.0, 200.0, 200.0,
                                   200.0, 200.0, 200.0, 200.0,
                                   ]]).cuda()
        self.uMin = torch.zeros_like(self.uMax).cuda()

        self.state_mean = ((self.xMax + self.xMin) / 2)
        self.state_std = ((self.xMax - self.xMin) / 2)
        self.input_mean = ((self.uMax + self.uMin) / 2)
        self.input_std = (self.uMax - self.uMin) / 2

        if not use_gpu:
            self.xMax = self.xMax.cpu().numpy()
            self.xMin = self.xMin.cpu().numpy()
            self.uMax = self.uMax.cpu().numpy()
            self.uMin = self.uMin.cpu().numpy()



if __name__ == "__main__":
    trial_dir = "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_left_hw/results/run_logs/base_0"
    model = LearnedModel_BalooHW(trial_dir)
