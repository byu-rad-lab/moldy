from lightning.pytorch.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Callback as pl_Callback
from numpy import ndarray
import matplotlib.pyplot as plt
import glob

def rk4(model:callable, x:ndarray, u:ndarray, dt:float) -> ndarray:
    """
    Runge-Kutta 4 integration method
    :param model: callable, model to integrate
    :param x: ndarray, state data of size (num_data_points, num_states)
    :param u: ndarray, input data of size (num_data_points, num_inputs)
    :param dt: float, time to integrate forward
    :return: ndarray, state derivative data of size (num_data_points, num_states)
    """
    F1 = model(x, u)
    F2 = model(x + dt / 2 * F1, u)
    F3 = model(x + dt / 2 * F2, u)
    F4 = model(x + dt * F3, u)
    return dt / 6 * (F1 + 2 * F2 + 2 * F3 + F4)


""" Currently have to inherit these callbacks due to known issue with ray tune call backs
            https://github.com/ray-project/ray/issues/33426#issuecomment-1477464856
"""
class _ModelCheckpoint(ModelCheckpoint, pl_Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _TuneReportCallback(TuneReportCallback, pl_Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def plot_loss_curve(config):
    val_loss = []
    metrics_path = glob.glob(f"{config['path']}results/run_logs/{config['run_name']}/**/**/metrics.csv")[-1]
    with open(metrics_path, "r") as f:
        idx = 0
        first = True
        for line in f:
            if first:
                idx = line[:-1].split(",").index("val_loss")
                first = False
            else:
                val_loss.append(float(line.split(",")[idx]))
    plt.figure()
    plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss for {config['run_name']}/{metrics_path.split('/')[-2]}")
    plt.savefig(f"{config['path']}results/run_logs/{config['run_name']}/val_loss_{metrics_path.split('/')[-2]}.png")
    plt.close()