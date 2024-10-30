import numpy as np
import torch
from typing import Callable, Tuple

class NonlinearEMPC:
    def __init__(
        self,
        forward_sim_func: Callable,
        cost_function: Callable,
        numStates: int,
        numInputs: int,
        umin: float,
        umax: float,
        horizon: int = 50,
        dt: float = 0.01,
        numSims: int = 500,
        numKnotPoints: int = 4,
        useGPU: bool = False,
        **kwargs,
    ):
        """Nonlinear Evolutionary Model Predictive Control Class

            For algorithmic details, see this paper:

            Hyatt, Phillip, and Marc D. Killpack.
            “Real-Time Nonlinear Model Predictive Control of Robots Using a Graphics Processing Unit.”
            IEEE Robotics and Automation Letters 5, no. 2 (April 2020): 1468–75.
            https://doi.org/10.1109/LRA.2020.2965393.

        :param forward_sim_func: Function that takes in state, input, and dt and returns the next state
        :param cost_function: Function that takes in state, input, goal state, and goal input and returns the cost
        :param numStates: Number of states
        :param numInputs: Number of inputs
        :param umin: Minimum input value, size (1, numInputs)
        :param umax: Maximum input value, size (1, numInputs)
        :param horizon: Number of time steps in the controller horizon
        :param dt: Time step of the controller
        :param numSims: Number of simulations to run for the control test
        :param numKnotPoints: Number of knot points to use in the genetic algorithm
        :param useGPU: Whether or not to use the GPU
        :param kwargs: Optional arguments
            :param kwargs["selection_method"]: str: The selection method to use in the genetic algorithm
            :param kwargs["tournament_size"]: int: The tournament size to use in the genetic algorithm
            :param kwargs["crossover_method"]: str: The crossover method to use in the genetic algorithm
            :param kwargs["mutation_probability"]: float: The mutation probability to use in the genetic algorithm
            :param kwargs["numParents"]: int: The number of parents to use in the genetic algorithm
            :param kwargs["numStrangers"]: int: The number of strangers to use in the genetic algorithm
        """
        torch.set_grad_enabled(False)
        self.model = forward_sim_func
        self.cost_function = cost_function
        self.numStates = numStates
        self.numInputs = numInputs
        self.umin = umin
        self.umax = umax
        self.uRange = umax - umin
        self.horizon = horizon
        self.numSims = numSims
        self.dt = dt
        self.numKnotPoints = numKnotPoints

        supported_selection_methods = ["elitism", "tournament"]
        self.selection_method = kwargs.get("selection_method", "elitism")
        if self.selection_method not in supported_selection_methods:
            raise ValueError(f"{self.selection_method} isn't supported.")
        self.tournament_size = kwargs.get("tournament_size", 5)
        self.crossover_method = kwargs.get("crossover_method", "knot_point")
        self.mutation_probability = kwargs.get("mutation_probability", 0.1)
        self.numParents = kwargs.get("numParents", 10)
        self.numStrangers = kwargs.get("numStrangers", 0)

        assert self.mutation_probability >= 0.0 and self.mutation_probability <= 1.0, "Mutation probability is out of bounds [0,1]"
        assert self.numParents + self.numStrangers < self.numSims, "Population split badly. Must have numParents + numStrangers < numSims."

        self.warmStart = False
        self.rng = np.random.default_rng(seed=42)
        if useGPU and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.U = np.zeros([self.numSims, self.numKnotPoints, self.numInputs])
        self.costs = np.zeros([self.numSims])
        self.costs_zero = torch.zeros([self.numSims,]).to(self.device)
        self.X = np.zeros([self.numSims, self.horizon, self.numStates])
        self.X_c_zero = torch.zeros([self.numSims, self.horizon, self.numStates]).to(self.device)

    def sample_new_commands(self, numSims: int) -> np.ndarray:
        """
        Get random input trajectories
        :param numSims: Number of random input trajectories to generate
        :return: Random input trajectories, size (numSims, knotPoints, numInputs)"""
        return (self.rng.random(size=(numSims, self.numKnotPoints, self.numInputs))
                    * self.uRange
                    + self.umin
                    )

    def select_parents(self) -> np.ndarray:
        """
        Select parents from the population
        :return: Parents, size (numParents, numKnotPoints, numInputs)
        """
        U_parents = np.zeros([self.numParents, self.numKnotPoints, self.numInputs])
        parentCount = 0
        if self.selection_method == "tournament":
            while parentCount < self.numParents:
                group_idxs = self.rng.integers(0, self.numSims - 1, self.tournament_size)
                tourney_winner_idx = group_idxs[np.argmin(self.costs[group_idxs])]
                U_parents[parentCount] = self.U[tourney_winner_idx]
                parentCount += 1
        elif self.selection_method == "stochastic_acceptance":
            minCost = np.min(self.costs)
            while parentCount < self.numParents:
                candidate = self.rng.integers(self.numSims)
                cost = self.costs[candidate]
                acceptProbability = minCost / cost
                if self.rng.random() < acceptProbability:
                    U_parents[parentCount] = self.U[candidate]
                    parentCount += 1
        else:   # elitism - just choose best numParents members of the population
            indices = self.costs.argsort().flatten()
            U_parents = self.U[indices[:self.numParents]]
        return U_parents

    def mate_parents(self, parents: np.ndarray) -> np.ndarray:
        """
        Mate parents to create children
        :param parents: Parents to mate, size (numParents, knotPoints, numInputs)
        :return: Children, size (numChildren, knotPoints, numInputs)
        """
        self.U[:self.numParents] = parents
        self.U[self.numParents:self.numParents+self.numStrangers] = self.sample_new_commands(self.numStrangers)

        numChildren = self.numSims - (self.numParents + self.numStrangers)
        if self.crossover_method == "knot_point":
            mates = self.rng.integers(0, self.numParents, (2, numChildren))
            parents0 = parents[mates[0]]
            parents1 = parents[mates[1]]

            inheritanceMask = self.rng.integers(2, size=parents0.shape)
            children = parents0 * inheritanceMask + parents1 * (1 - inheritanceMask)
        else:
            raise ValueError(f"{self.crossover_method} isn't a supported crossover method.")
        
        return children
        
    def mutate_children(self, children: np.ndarray, mutation_noise: float) -> None:
        """
        Mutate the children with Gaussian noise
        :param children: Children to mutate, size (numChildren, knotPoints, numInputs)
        :param mutation_noise: Maximum amount that a child will be mutated (+- mutation_noise)
        :return: None
        """
        numChildren = self.numSims - (self.numParents + self.numStrangers)
        random = self.rng.uniform(0, 1, size=[numChildren])
        mutationMask = random > (1 - self.mutation_probability)
        numMutated = np.sum(mutationMask)

        mutation = self.rng.normal(0., 1, size=[numMutated, self.numKnotPoints, self.numInputs]) * mutation_noise
        children[mutationMask] += mutation

        children = np.clip(children, self.umin, self.umax)
        return children

    def get_us_from_U(self) -> np.ndarray:
        """
        Get the inputs for the trajectory based on knotPoints
        :return: Inputs for the given time step, size (numSims, horizon, numInputs)
        """

        if self.numKnotPoints == 1:
            ui = self.U.repeat(self.horizon, axis=1)
        else:
            segmentLength = float(self.horizon - 1) / (self.numKnotPoints - 1)
            i = np.arange(self.horizon)
            knot = (i / segmentLength).astype(int)

            Ulow = self.U[:, knot]
            knot[-1] -= 1
            Uhigh = self.U[:, knot + 1]
            ratio = (i % segmentLength) / segmentLength
            ui = Uhigh + (1-ratio.reshape(1,self.horizon,1))*Ulow
        ui = np.clip(ui, self.umin, self.umax)
        return ui
    
    def get_costs_cpu(self, x0:np.ndarray, xgoal:np.ndarray, ugoal:np.ndarray, uprev:np.ndarray=None) -> None:
        """
        Forward simulate the population through the horizon and assign each member a cost using the cpu
        :param x0: Initial state of the system, size (1, numStates)
        :param xgoal: Goal state of the system, size (1, numStates)
        :param ugoal: Goal input of the system, size (1, numInputs)
        :param uprev: Previous input of the system, size (1, numInputs)
        :return: None

        Updates self.costs and self.X directly
        """
        self.costs = np.zeros([self.numSims])
        self.X[:,0] = x0
        U = self.get_us_from_U()
        U[:, 0] = uprev

        for i in range(0, self.horizon - 1):
            if self.base_model is None:
                self.X[:, i+1] = self.model(self.X[:, i], U[:,i], self.dt)
            else:
                self.X[:, i+1] = self.model(self.X[:, i], U[:,i], self.dt, self.base_model, i)
            self.costs += self.cost_function(self.X[:, i], U[:,i], xgoal, ugoal, prev_u=U[:,i-1])

        FinalCosts = self.cost_function(self.X[:, self.horizon - 1], U[:,i-1], xgoal, ugoal, prev_u=U[:,i-2], final_timestep=True)
        self.costs += FinalCosts

    def get_costs_gpu(self, x0:np.ndarray, xgoal:np.ndarray, ugoal:np.ndarray, uprev:np.ndarray=None) -> None:
        """
        Forward simulate the population through the horizon and assign each member a cost.
        :param x0: Initial state of the system, size (1, numStates)
        :param xgoal: Goal state of the system, size (1, numStates)
        :param ugoal: Goal input of the system, size (1, numInputs)
        :param uprev: Previous input of the system, size (1, numInputs)
        :return: None

        Updates self.costs and self.X directly
        """
        xgoal_c = torch.from_numpy(xgoal).to(self.device).float()
        ugoal_c = torch.from_numpy(ugoal).to(self.device).float()
        uprev = torch.from_numpy(uprev).to(self.device).float()
        U_c = torch.from_numpy(self.get_us_from_U()).to(self.device).float()
        costs = self.costs_zero.clone()

        U_c[:, 0] = uprev.repeat(self.numSims,1)
        X_c = self.X_c_zero.clone()
        x0_c = torch.tensor(x0).to(self.device).float()
        X_c[:, 0] = x0_c.repeat(self.numSims,1)

        for i in range(0, self.horizon - 1):
            X_c[:, i+1] = torch.as_tensor(self.model(X_c[:, i], U_c[:,i], self.dt)).to(self.device)
            costs += self.cost_function(X_c[:,i], U_c[:,i], xgoal_c, ugoal_c, prev_u=U_c[:,i-1])

        costs += self.cost_function(X_c[:, self.horizon - 1], U_c[:,i-1], xgoal_c, ugoal_c, prev_u=U_c[:,i-2], final_timestep=True)

        self.X = X_c.cpu().numpy()
        self.costs = costs.cpu().numpy()

    def get_best_inputs(self) -> np.ndarray:
        """
        Get the next input to be applied (the one corresponding to the lowest cost)
        :return: Best inputs, size (1, numInputs)
        """
        best_trajectory = np.nanargmin(self.costs)
        return self.U[best_trajectory]
    
    def get_planned_path(self) -> np.ndarray:
        """
        Get the best trajectory that will be returned by nempc
        :return: Best trajectory, size (horizon, numStates)
        """
        best_trajectory = np.nanargmin(self.costs)
        return self.X[best_trajectory]

    def solve_for_next_u(self, x0:np.ndarray, xgoal:np.ndarray, ugoal:np.ndarray, ulast:np.ndarray, mutation_noise:float=0.99, base_model=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for the next input to apply to the system
        :param x0: Initial state of the system, size (1, numStates)
        :param xgoal: Goal state of the system, size (1 or horizon, numStates)
        :param ugoal: Goal input of the system, size (1 or horizon, numInputs)
        :param ulast: Previous input of the system, size (1, numInputs)
        :param mutation_noise: Mutation noise to use for the genetic algorithm
        :param base_model: only necessary if using a mujoco model for simulation
        
        :return: Tuple[np.ndarray, np.ndarray]:
            u: Input to apply to the system, size (1, numInputs)
            path: Best trajectory that will be returned by nempc, size (horizon, numStates)
        """
        self.base_model = base_model
        if not self.warmStart:
            self.U = self.sample_new_commands(self.numSims)
            self.warmStart = True
        else:
            U_parents = self.select_parents()
            children = self.mate_parents(U_parents)
            self.U[self.numParents + self.numStrangers:] = self.mutate_children(children, mutation_noise)
        
        if self.device == "cpu":
            self.get_costs_cpu(x0, xgoal, ugoal, uprev=ulast)
        else:
            self.get_costs_gpu(x0, xgoal, ugoal, uprev=ulast)

        u = self.get_best_inputs()
        path = self.get_planned_path()
        return u, path