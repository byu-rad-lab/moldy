from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass
from moldy.case_studies.inverted_pendulum.model_ip import InvertedPendulum

class InvertedPendulumLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        super().__init__(config)
        self.system = InvertedPendulum()