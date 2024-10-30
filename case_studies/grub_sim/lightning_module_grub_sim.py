from moldy.model.LightningModuleBaseClass import LightningModuleBaseClass
from moldy.case_studies.grub_sim.model_grub_sim import GrubSim


class GrubSimLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        if config.get("grub_params", None) is not None:
            self.system = GrubSim(**config["grub_params"])
        else:
            vary_params = config.get("vary_params_percent", None)
            if vary_params is None: # If it is none, then it is an old model using old parameters
                self.system = GrubSim(old_params=True)
            else:
                self.system = GrubSim(vary_params_percent=vary_params)
                
            grub_params = {
                "mass": self.system.m,
                "stiffness": self.system.stiffness,
                "damping": self.system.damping,
                "pressure_resp_coeff": self.system.alpha,
                "h": self.system.h,
                "r": self.system.r,
            }
            config["grub_params"] = grub_params

        super().__init__(config)


