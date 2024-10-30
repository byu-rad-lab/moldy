import torch
import numpy as np

from moldy.model.LearnedModel import LearnedModel
from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim
from moldy.case_studies.baloo_sim.lightning_module_baloo_sim import BalooSimLightningModule

class LearnedModel_BalooSim(LearnedModel, BalooSim):
    def __init__(self, trial_dir=None, **kwargs):

        super().__init__(LightningModule=BalooSimLightningModule, trial_dir=trial_dir, **kwargs)
        self.name = "learned_baloo_left"

        # self.output_max = torch.from_numpy(np.load("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/train_output_max.npy")).float().cuda()
        # self.output_min = torch.from_numpy(np.load("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/data/train_output_min.npy")).float().cuda()
        
    # def forward_simulate_dt(self, x:torch.Tensor, u:torch.Tensor, dt:float=0.01) -> torch.Tensor:
        # """
        # Forward simulate the model using the neural network
        # :param x: torch.Tensor, state data of size (num_data_points, sequence_len, num_states)
        # :param u: torch.Tensor, input data of size (num_data_points, sequence_len, num_inputs)
        # :param dt: float, time step
        # :return: torch.Tensor, state data of size (num_data_points, num_states)

        # dt is not used because the models learn a constant dt. This could be changed in the future to be able
        # predict using a variable dt.
        # """

        # # x = torch.clamp(x, self.xMin, self.xMax)
        # # u = torch.clamp(u, self.uMin, self.uMax)

        # if self.learn_mode == "delta_x":
        #     delta_x = self.calc_state_derivs(x, u)
        #     x += delta_x

        #     return x
        

if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import mujoco
    import mujoco.viewer

    trial_dir="/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/results/run_logs/test1"

    learned_system = LearnedModel_BalooSim(trial_dir)
    x = learned_system.generate_random_state()
    u = torch.from_numpy(learned_system.generate_random_command()).cuda()

    dt = 0.01
    sim_time = 2.0
    horizon = int(sim_time / dt)

    x_history = np.zeros((horizon, learned_system.numStates))
    u_history = np.zeros((horizon, learned_system.numInputs))

    # with mujoco.viewer.launch_passive(learned_system.mujoco_model, learned_system.data) as viewer:
    for i in range(0, horizon):
        x = learned_system.forward_simulate_dt(x, u, dt)

        x_history[i, :] = x.detach().cpu().numpy().flatten()
        u_history[i, :] = u.detach().cpu().numpy().flatten()

            # if i % 5 == 0:
            #     learned_system.visualize(viewer)
            #     plt.pause(0.1)

    learned_system.plot_history(x_history, u_history, xgoal=np.zeros((horizon, learned_system.numStates)))
