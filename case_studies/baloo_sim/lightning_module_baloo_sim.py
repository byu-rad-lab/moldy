from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass
from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim


class BalooSimLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        super().__init__(config)
        self.system = BalooSim()
