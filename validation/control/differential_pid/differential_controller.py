import numpy as np


class DifferentialController:
    def __init__(self, kp:np.ndarray, kd:np.ndarray, ki:np.ndarray, min:float, max:float, sigma:float=0.05, Ts:float=0.01) -> None:
        """
        Differential controller (PD Control)
        :param kp: np.ndarray, proportional gain, size (num_states, num_states)
        :param kd: np.ndarray, derivative gain, size (num_states, num_states)
        :param ki: np.ndarray, integral gain, size (num_states, num_states)
        :param min: float, minimum command
        :param max: float, maximum command
        :param sigma: float, noise to add to the command, default 0.05
        :param Ts: float, timestep, default 0.01
        """
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.min = min
        self.max = max
        self.sigma = sigma
        self.Ts = Ts
        self.beta = (2.0*self.sigma - self.Ts) / (2.0*self.sigma + self.Ts)

        self.x_d1 = 0.0
        self.xdot = 0.0
        self.errorInt = 0.0
        self.error_d1 = 0.0

    def calc_delta_u(self, x:np.ndarray, xgoal:np.ndarray) -> np.ndarray:
        """
        Calculate the command
        :param x: np.ndarray, state, size (num_states,)
        :param xgoal: np.ndarray, state goal, size (num_states,)

        :return: np.ndarray, command, size (num_states,)
        """
        error = xgoal - x
        self.xdot = self.beta * self.xdot + (1 - self.beta) * (x - self.x_d1) / self.Ts
        self.errorInt += (error + self.error_d1) * self.Ts

        u = self.kp @ error - self.kd @ self.xdot + self.ki @ self.errorInt

        self.error_d1 = error
        self.x_d1 = x

        return u
    
    def calc_final_u(self, delta_u:np.ndarray) -> np.ndarray:
        """
        Calculate the command from the delta command

        :param delta_u: np.ndarray, delta command, size (num_inputs,)
        :return: np.ndarray, command, size (num_inputs,)
        """
        u_final = np.ones((1,2*delta_u.shape[0])) * (self.max - self.min)/2

        for i in range(0, delta_u.shape[0]*2, 2):
            u_final[:,i] += delta_u[i//2]
            u_final[:,i+1] -= delta_u[i//2]

        u_final = np.clip(u_final, self.min, self.max)
        return u_final
    
if __name__=="__main__":
    x = np.array([3.2, 4.5])
    xgoal = np.array([3.0, 4.0])

    kp = np.diag([1.0, 1.0])
    kd = np.diag([1.0, 1.0])
    ki = np.diag([1.0, 1.0])

    min = 0.0
    max = 300.0

    controller = DifferentialController(kp, kd, ki, min, max)
    u_delta = controller.calc_delta_u(x, xgoal)
    u_final = controller.calc_final_u(u_delta)

    print(u_delta)
    print(u_final)

    print(u_delta.shape)
    print(u_final.shape)