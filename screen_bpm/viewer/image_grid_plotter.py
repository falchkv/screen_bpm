import multiprocessing as mp
import time

from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy

# Fixing random state for reproducibility
numpy.random.seed(19680801)


class ProcessPlotter:
    def __init__(self, reference_uvs=None, interval=100):
        self.interval = interval

        if reference_uvs is not None:
            self.reference_uvs = reference_uvs
        else:
            self.reference_uvs = {}

        self.nrow, self.ncol = (2, 2)  # This should be dynamically determined
        self.image_plots = [
            [None for i in range(self.ncol)] for j in range(self.nrow)
        ]

        self.cmap = 'gray'
        self.reference_color = 'tab:red'
        self.current_color = 'tab:green'

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            plot_dict = self.pipe.recv()
            if plot_dict is None:
                self.terminate()
                return False
            else:
                #self.fig.suptitle(
                #    'count {:05d}'.format(plot_dict['count_id']))

                for k, key in enumerate(plot_dict['images']):
                    i = int(k / self.ncol)
                    j = int(numpy.mod(k, self.ncol))
                    #self.image_plots[i][j].set_data(plot_dict['images'][key])

                    cax = self.ax[i, j]
                    self.image_plots[i][j] = cax.imshow(
                        plot_dict['images'][key], cmap=self.cmap, vmin=0, vmax=255
                    )
                    cax.set_title(key)

                    # plot reference indicator
                    if key in self.reference_uvs:
                        y, x = self.reference_uvs[key]
                        cax.plot(x, y, 'x', color=self.reference_color)

                    # plot current indicator
                    if key in self.reference_uvs:
                        y, x = plot_dict['uv_points'][key]
                        cax.plot(x, y, 'x', color=self.current_color)

                    # TMP: print uv difference
                    if key in self.reference_uvs:
                        y, x = plot_dict['uv_points'][key]
                        y_ref, x_ref = self.reference_uvs[key]
                        y_diff = y - y_ref
                        x_diff = x - x_ref
                        print(f'{key} diff: {y_diff}, {x_diff}')

        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()



        return True

    def __call__(self, pipe):
        self.pipe = pipe
        self.fig, self.ax = plt.subplots(self.nrow, self.ncol)
        for i in range(self.nrow):
            for j in range(self.ncol):
                cax = self.ax[i, j]
                self.image_plots[i][j] = cax.imshow(numpy.random.rand(1, 1))

        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()


class PltGridPlotter:
    def __init__(self, reference_uvs=None, interval=100):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(
            reference_uvs=reference_uvs, interval=interval)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, plot_dict, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(plot_dict)

def main():
    pl = PltGridPlotter()
    for _ in range(10000):
        image_stack = numpy.random.rand(3, 100, 100)
        plot_dict = {
            'count_id': int(numpy.random.rand()*100),
            'images': {
                'LM2': image_stack[0],
                'LM3': image_stack[1],
                'LM4': image_stack[2],
            }
        }
        pl.plot(plot_dict)
        time.sleep(0.3)
    pl.plot(finished=True)


if __name__ == '__main__':
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")
    main()