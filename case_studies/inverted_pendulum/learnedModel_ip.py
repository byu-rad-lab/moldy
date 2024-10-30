from moldy.model.LearnedModel import LearnedModel
from moldy.case_studies.inverted_pendulum.lightning_module_ip import InvertedPendulumLightningModule
from moldy.case_studies.inverted_pendulum.model_ip import InvertedPendulum


class LearnedModel_InvertedPendulum(LearnedModel, InvertedPendulum):
    def __init__(self, trial_dir:str=None, **kwargs:dict):
        super().__init__(LightningModule=InvertedPendulumLightningModule, trial_dir=trial_dir, **kwargs)
        self.name = "LearnedModel_InvertedPendulum"


if __name__ == "__main__":
    import torch
    import numpy as np

    trial_dir = "case_studies/inverted_pendulum/results/best_models/opt_cosine_4"

    learned_system = LearnedModel_InvertedPendulum(trial_dir)
    x = (learned_system.generate_random_state()).cuda()
    u = torch.from_numpy(learned_system.generate_random_command()).cuda()
    x = torch.zeros((1, 2)).cuda()
    x[:,1] = 0.001
    u = torch.zeros((1, 1)).cuda()

    dt = 0.01
    sim_time = 10
    horizon = int(sim_time / dt)

    x_history = np.zeros((horizon, learned_system.numStates))
    u_history = np.zeros((horizon, learned_system.numInputs))

    for i in range(0, horizon):
        x = learned_system.forward_simulate_dt(x, u, dt)

        x_history[i, :] = x.detach().cpu().numpy().flatten()
        u_history[i, :] = u.detach().cpu().numpy().flatten()

        if i % 5 == 0:
            learned_system.visualize(x, u)

    learned_system.plot_history(x_history, u_history, xgoal=np.zeros((1,learned_system.numStates)))
