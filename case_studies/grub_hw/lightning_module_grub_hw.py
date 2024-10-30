from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass
from moldy.case_studies.grub_sim.model_grub_sim import GrubSim

class GrubHWLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        super().__init__(config)
        self.system = GrubSim()
        self.system.name = "grub_hw"