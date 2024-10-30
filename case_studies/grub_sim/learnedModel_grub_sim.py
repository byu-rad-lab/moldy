from moldy.model.LearnedModel import LearnedModel
from moldy.case_studies.grub_sim.lightning_module_grub_sim import GrubSimLightningModule
from moldy.case_studies.grub_sim.model_grub_sim import GrubSim


class LearnedModel_GrubSim(LearnedModel, GrubSim):
    def __init__(self, trial_dir:str=None, **kwargs):
        super().__init__(LightningModule=GrubSimLightningModule, trial_dir=trial_dir, **kwargs)
        self.name = "LearnedModel_GrubSim"


if __name__ == "__main__":
    import torch
    import numpy as np

    trial_dir = "moldy/case_studies/grub_sim/results/best_models/rmse_4"

    learned_system = LearnedModel_GrubSim(trial_dir)
    x = torch.from_numpy(learned_system.generate_random_state()).cuda()
    u = torch.from_numpy(learned_system.generate_random_command()).cuda()

    dt = 0.01
    sim_time = 2.0
    horizon = int(sim_time / dt)

    x_history = np.zeros((horizon, learned_system.numStates))
    u_history = np.zeros((horizon, learned_system.numInputs))

    for i in range(0, horizon):
        x = learned_system.forward_simulate_dt(x, u, dt)

        x_history[i, :] = x.detach().cpu().numpy().flatten()
        u_history[i, :] = u.detach().cpu().numpy().flatten()

    print(x_history[-1, :])
    print(u_history[-1, :])

    learned_system.plot_history(
        x_history, u_history, xgoal=np.zeros((1, learned_system.numStates))
    )
