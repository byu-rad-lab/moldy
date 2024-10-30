import random
import numpy as np
import matplotlib.pyplot as plt
import os 

from moldy.model.Model import Model

def generate_square_wave(num_waves:int, num_steps:int, min:np.ndarray, max:np.ndarray) -> np.ndarray:
    """
    Generate a square wave
    :param num_waves: int, number of waves to generate 
    :param min: np.ndarray, minimum values of the square wave
    :param max: np.ndarray, maximum values of the square wave
    :param num_steps: int, number of steps in the square wave
    :return: np.ndarray, square wave data of size (num_steps, num_waves)
    """
    wave_length_range = (100, 1000)

    square_wave = np.zeros((num_steps, num_waves))
    i = 0
    while i < num_steps-1:
        wave_length = np.random.randint(wave_length_range[0], wave_length_range[1])
        if wave_length + i >= num_steps:
            wave_length = num_steps - i

        wave = np.random.randint(min, max, size=(1, num_waves))
        square_wave[i:i+wave_length] = wave
        i += wave_length

    return square_wave
class SinwaveTrajectoryGenerator():
    def __init__(self, system: Model):
        """
        Base class for Making randomizd sinusoidal waves
        """
        assert (
            type(system) is not None
        ), "System type is None. Check that the System class is passed in correctly."
        
        self.system = system
        
    def coef_randomizer(self,complexity=1,amp=1):
                coefficients = np.zeros((int(complexity), 3))
                omega_bounds = [0, 0.25] #[0, 1] 
                phi_bounds = [0, 2 * np.pi] 
                for i in range(int(complexity)):
                    # coefficients[i, 0] = (amp[0][0]-amp[0][0]/100)/complexity
                    coefficients[i, 0] = 1/complexity
                    coefficients[i, 1] = np.random.uniform(omega_bounds[0], omega_bounds[1])
                    coefficients[i, 2] = np.random.uniform(phi_bounds[0], phi_bounds[1])
                return coefficients

    def sinwavegen(self, coef_matrix, duration=1.0, sampling_rate=25):
                self.time = np.linspace(0, duration, int(duration * sampling_rate))
                sinwave_array = np.zeros((coef_matrix.shape[0], self.size*sampling_rate)) #[]

                for i in range(coef_matrix.shape[0]):
                    amplitude, frequency, phase = coef_matrix[i]
                    sinwave = amplitude * np.sin(2 * np.pi * frequency * self.time + phase)
                    sinwave_array[i] = sinwave
                return sinwave_array
                
    def generate_sin_data(self, size: int,save_path=None, display:bool=False) -> np.ndarray:
        self.size = size
        num_waves = 2 #self.system.numInputs
        self.save_path = save_path
        amplitude = np.array([self.system.xMax[0][6:8] - self.system.xMin[0][6:8]]) #self.system.uMax-self.system.uMin 
        duration, sampling_rate = size,25
        complexity = np.random.randint(3,5)  
        command_array = np.zeros((num_waves, size*sampling_rate))

        for j in range(num_waves):
            coefficients = self.coef_randomizer(complexity,amplitude/2)
            sinwaves = self.sinwavegen(coefficients, duration, sampling_rate)
            sinwave_sum = sum(sinwaves)
            sinwave_sum = sinwave_sum  #+ (amplitude[0][0]/2)
            command_array[j] = sinwave_sum

        if display:
            plt.figure(figsize=(10, 6))
            plt.plot(self.time, command_array.T)
            plt.title('Sinwaves')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.show()
        
        if save_path is not None:
            command_array = np.vstack([np.zeros((6, size*sampling_rate)), command_array])
            np.save(self.save_path + f'/{duration}s_sin_commands.npy', command_array.T)

        return command_array.T


if __name__ == "__main__":
    from moldy.case_studies.grub_sim.model_grub_sim import GrubSim
    grubSim = GrubSim()
    save_path = '/home/student/catkin_ws/src/moldy/case_studies/grub_hw/data/sin_joint_cmds'
    sinwavegenerator = SinwaveTrajectoryGenerator(grubSim)
    sinwavegenerator.generate_sin_data(3600,save_path=save_path,display=True)
    