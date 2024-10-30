import numpy as np
import yaml
import mujoco

from moldy.case_studies.baloo_sim.model_baloo_sim import BalooSim
from moldy.case_studies.baloo_sim.model.generate_baloo_sim import generateBalooXML

class BalooSimSysIdWrapper(BalooSim):
    def __init__(self, **kwargs:dict):
        """
        Params that we are fitting (for each joint):
            - mass
            - radius
            - bellows_radius
            - bellows_effective_area
            - lumped_stiffness
            - lumped_damping
            - pressure_time_constant
                """
        super().__init__(**kwargs)

        self.max_vals = {
            'large_mass': 22.0,
            'large_radius': .25,
            'large_bellows_radius': 0.1,
            'large_bellows_effective_area': 0.025,
            'large_lumped_stiffness': 135,
            'large_lumped_damping': 85,
            'large_pressure_time_constant': 0.25,

            'medium_mass': 7.5,
            'medium_radius': 0.25,
            'medium_bellows_radius': 0.05,
            'medium_bellows_effective_area': 0.05,
            'medium_lumped_stiffness': 75,
            'medium_lumped_damping': 20,
            'medium_pressure_time_constant': 0.25,

            'small_mass': 7.5,
            'small_radius': 0.25,
            'small_bellows_radius': 0.1,
            'small_bellows_effective_area': 0.05,
            'small_lumped_stiffness': 45,
            'small_lumped_damping': 30,
            'small_pressure_time_constant': 0.5
        }
        self.min_vals = {
            'large_mass': 18.0,
            'large_radius': 0.10,
            'large_bellows_radius': 0.04,
            'large_bellows_effective_area': 0.01,
            'large_lumped_stiffness': 110,
            'large_lumped_damping': 75,
            'large_pressure_time_constant': 0.05,

            'medium_mass': 2.5,
            'medium_radius': 0.075,
            'medium_bellows_radius': 0.01,
            'medium_bellows_effective_area': 0.01,
            'medium_lumped_stiffness': 50,
            'medium_lumped_damping': 1,
            'medium_pressure_time_constant': 0.05,

            'small_mass': 2.5,
            'small_radius': 0.1,
            'small_bellows_radius': 0.05,
            'small_bellows_effective_area': 0.01,
            'small_lumped_stiffness': 20,
            'small_lumped_damping': 10,
            'small_pressure_time_constant': 0.1
        }

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

        with open("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_params.yaml", "w") as f:
            yaml.dump(real_params, f)
            yaml.dump({"general": {"joint_height": 0.2, "max_pressure":400000, "arm_angle":67.5}}, f)
            yaml.dump({"large_bend_limit":90, "medium_bend_limit":110, "small_bend_limit":110}, f)
            # yaml.dump({'large_mass': 2.653,
            #             'large_radius': 0.125,
            #             # 'large_bellows_radius': 0.075,
            #             'medium_mass': 1.326,
            #             'medium_radius': 0.1,
            #             # 'medium_bellows_radius': 0.06,
            #             'small_mass': 1.326,
            #             'small_radius': 0.08,
            #             # 'small_bellows_radius': 0.04
            #             }, f)

        generateBalooXML("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_baloo.xml",
            "/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_params.yaml")
        
        self.setup_mujoco("/home/daniel/catkin_ws/src/moldy/case_studies/baloo_sim/model/sys_id_baloo.xml")
  
    def calc_state_derivs(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        """ Calculate the state derivatives
        :param x: state array, np.ndarray of size (num_data_points, num_states)
        :param u: command array, np.ndarray of size (num_data_points, num_inputs)
        :return: state derivative array, np.ndarray of size (num_data_points, num_states)
        """
        u = np.clip(u, self.uMin, self.uMax) 
        dt = 0.01

        xdot = np.zeros(x.shape)

        for i in range(self.numSims):
            # self.set_state(x[i].reshape(1,self.numStates))
            for state in range(x.shape[0]):
                self.set_pcmd(u[state].reshape(1,self.numInputs), i=i)

                for j in range(int(dt / self.mujoco_dt)):
                    mujoco.mj_step(self.mujoco_model, self.data[i])

                xdot[state] = (self.get_state(i) - x[state])

        return xdot

    def curvefit_wrapper(self, x:np.ndarray, u:np.ndarray,
                        large_mass:float, large_radius:float, 
                        large_bellows_radius:float, 
                        large_bellows_effective_area:float, large_lumped_stiffness:float, large_lumped_damping:float, large_pressure_time_constant:float,
                        medium_mass:float, medium_radius:float, 
                        medium_bellows_radius:float, 
                        medium_bellows_effective_area:float, medium_lumped_stiffness:float, medium_lumped_damping:float, medium_pressure_time_constant:float,
                        small_mass:float, small_radius:float, 
                        small_bellows_radius:float, 
                        small_bellows_effective_area:float, small_lumped_stiffness:float, small_lumped_damping:float, small_pressure_time_constant:float,
                         ) -> np.ndarray:
        """
        Wrapper function for lmfit to use to fit the system parameters
        :param x: state array, np.ndarray of size (num_data_points, 24)
        :param u: command array, np.ndarray of size (num_data_points, 12)
        :param kwargs: dictionary of scaled parameters passed in from lmfit
        """
        params = {
            'large_mass': large_mass,
            'large_radius': large_radius,
            'large_bellows_radius': large_bellows_radius,
            'large_bellows_effective_area': large_bellows_effective_area,
            'large_lumped_stiffness': large_lumped_stiffness,
            'large_lumped_damping': large_lumped_damping,
            'large_pressure_time_constant': large_pressure_time_constant,
            'medium_mass': medium_mass,
            'medium_radius': medium_radius,
            'medium_bellows_radius': medium_bellows_radius,
            'medium_bellows_effective_area': medium_bellows_effective_area,
            'medium_lumped_stiffness': medium_lumped_stiffness,
            'medium_lumped_damping': medium_lumped_damping,
            'medium_pressure_time_constant': medium_pressure_time_constant,
            'small_mass': small_mass,
            'small_radius': small_radius,
            'small_bellows_radius': small_bellows_radius,
            'small_bellows_effective_area': small_bellows_effective_area,
            'small_lumped_stiffness': small_lumped_stiffness,
            'small_lumped_damping': small_lumped_damping,
            'small_pressure_time_constant': small_pressure_time_constant
        }

        x = (x * ((self.xMax - self.xMin) / 2)) + ((self.xMax + self.xMin) / 2)
        u = (u * ((self.uMax - self.uMin) / 2)) + ((self.uMax + self.uMin) / 2)

        self.set_params(params)
        x_dot = self.calc_state_derivs(x, u)

        x_dot = (x_dot - ((self.xMax + self.xMin) / 2)) / ((self.xMax - self.xMin) / 2)

        return x_dot


if __name__=="__main__":
    sys = BalooSimSysIdWrapper()
    params = {
        'large_mass': 2.653,
        'large_radius': 0.125,
        'large_bellows_radius': 0.075,
        'large_bellows_effective_area': 0.004,
        'large_lumped_stiffness': 135,
        'large_lumped_damping': 10,
        'large_pressure_time_constant': 0.08,
        'medium_mass': 1.326,
        'medium_radius': 0.1,
        'medium_bellows_radius': 0.06,
        'medium_bellows_effective_area': 0.002,
        'medium_lumped_stiffness': 50,
        'medium_lumped_damping': 7.5,
        'medium_pressure_time_constant': 0.1,
        'small_mass': 1.326,
        'small_radius': 0.08,
        'small_bellows_radius': 0.04,
        'small_bellows_effective_area': 0.002,
        'small_lumped_stiffness': 30,
        'small_lumped_damping': 7.5,
        'small_pressure_time_constant': 0.2
    }

    sys.set_new_params(params)