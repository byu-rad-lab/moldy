import numpy as np
from moldy.case_studies.grub_sim.model_grub_sim import GrubSim

class GrubSimSysIdWrapper(GrubSim):
    def __init__(self, **kwargs:dict):
        super().__init__(**kwargs)

        self.xdotMax = np.array([[340.0,340.0,340.0,340.0,20.0,20.0,2.0,2.0]])
        self.xdotMin = np.array([[-250.0,-250.0,-250.0,-250.0,-20.0,-20.0,-2.0,-2.0]])

        self.max_vals =  {'stiffness': 50, 'damping': 50, 'alpha': 50}
        self.min_vals = {'stiffness': 0, 'damping': 0, 'alpha': 1}
        
    def scale_xdot(self, xdot:np.ndarray):
        """
        Unused function, but could be useful for scaling the truth data the same as the forward simulated data
        :param xdot: truth state data, np.ndarray of size (num_data_points, num_states)
        """
        # scale each column of xdot to be between -1 and 1
        xdot_scaled = np.zeros(xdot.shape)
        for i in range(xdot.shape[1]):
            min = self.xdotMin[0,i]
            max = self.xdotMax[0,i]
            xdot_scaled[:,i] = 2 * (xdot[:,i] - min) / (max - min) - 1
        
        return xdot_scaled

    def unscale_params(self, scaled_params:dict):
        """
        Params are scaled to make it easier for lmfit to converge
        :param scaled_params: dictionary of scaled parameters passed in from lmfit
        """
        params = {}
        for key in scaled_params:
            params[key] = float(((scaled_params[key] + 1) / 2) * (self.max_vals[key] - self.min_vals[key]) + self.min_vals[key])
        return params

    def set_params(self, params:dict):
        """
        Sets the system parameters, unscaling them first based on min and max values
        :param params: dictionary of scaled parameters passed in from lmfit
        """
        real_params = self.unscale_params(params)

        self.stiffness = real_params['stiffness']
        self.damping = real_params['damping']
        self.alpha = real_params['alpha']

    def curvefit_wrapper(self, x, u, stiffness, damping, alpha):
        """
        Wrapper function for lmfit to use to fit the system parameters
        :param x: state array, np.ndarray of size (num_data_points, 8)
        :param u: command array, np.ndarray of size (num_data_points, 4)
        :param stiffness: float, stiffness parameter
        :param damping: float, damping parameter
        :param alpha: float, alpha parameter
        """
        self.set_params({'stiffness':stiffness, 'damping':damping, 'alpha':alpha})
        x_dot = self.calc_state_derivs(x, u)
        return x_dot
