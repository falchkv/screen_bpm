import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)




class ProcessPlotter:
    def __init__(self):
        self.beam_x = []
        self.beam_y = []
        self.beam_z = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            plottables = self.pipe.recv()
            if plottables is None:
                self.terminate()
                return False
            else:
                self.beam_x = plottables['beam_xy'][:, 0]
                self.beam_y = plottables['beam_xy'][:, 1]
                self.beam_z = plottables['zs_of_interest']
                # ZX
                cax = self.ax[0]
                cax.plot(self.beam_y, self.beam_x, '-ro')

                # ZX
                cax = self.ax[1]
                cax.plot(self.beam_z, self.beam_y, '-ro')

        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        self.pipe = pipe
        self.fig, self.ax = plt.subplots(2, 1)
        # ZX
        cax = self.ax[0]
        cax.set_title('ZX')
        cax.grid()
        #cax.set_xlabel('z / m')
        cax.set_ylabel('x / mm')

        # ZY
        cax = self.ax[1]
        cax.set_title('ZY')
        cax.grid()
        cax.set_xlabel('z / m')
        cax.set_ylabel('y / mm')

        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()

class NBPlot:
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, plotables, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(plotables)

def main():
    pl = NBPlot()
    for _ in range(10000):
        plotables = {
            'beam_xy': np.random.rand(4, 2),
            'beam_angles': np.random.rand(4, 2),
            'zs_of_interest': np.random.rand(4),
            'uv_points': np.random.rand(4, 2),
        }
        pl.plot(plotables)
        time.sleep(0.3)
    pl.plot(finished=True)

if __name__ == '__main__':
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")
    main()