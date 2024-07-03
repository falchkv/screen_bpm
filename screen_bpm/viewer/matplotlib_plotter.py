import multiprocessing as mp
import time

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy

# Fixing random state for reproducibility
numpy.random.seed(19680801)


class ProcessPlotter:
    def __init__(self, reference_uvs=None, interval=100):
        self.reference_uvs = reference_uvs
        self.interval = interval
        self.beam_y = []
        self.beam_z = []
        self.line_zy = matplotlib.lines.Line2D
        self.line_zx = matplotlib.lines.Line2D
        self.image_plots = []
        self.texts = None
        self.text_offset = -9

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            plot_dict = self.pipe.recv()
            if plot_dict is None:
                self.terminate()
                return False
            else:
                self.beam_x = plot_dict['beam_xy'][:, 0]
                self.beam_y = plot_dict['beam_xy'][:, 1]
                self.beam_z = plot_dict['zs_of_interest']

                x_unit = 1e-3  # the unit given in m
                y_unit = 1e-3  # the unit given in m
                z_unit = 1.0  # the unit given in m
                # ZX
                self.line_zx.set_ydata(self.beam_x / x_unit)
                self.line_zx.set_xdata(self.beam_z / z_unit)
                self.points_zx.set_xdata(plot_dict['screen_xyz'][2] / z_unit)
                self.points_zx.set_ydata(plot_dict['screen_xyz'][0] / x_unit)

                # ZY
                self.line_zy.set_ydata(self.beam_y / y_unit)
                self.line_zy.set_xdata(self.beam_z / z_unit)
                self.points_zy.set_xdata(plot_dict['screen_xyz'][2] / z_unit)
                self.points_zy.set_ydata(plot_dict['screen_xyz'][1] / y_unit)

                # draw z_labels
                if 'z_labels' in plot_dict:
                    if self.texts is None:
                        self.texts = [self.ax[1].text(z, self.text_offset, label) for z, label in zip(plot_dict['zs_of_interest'], plot_dict['z_labels'])]


                # reference lines
                if 'reference' in plot_dict:
                    ref_dict = plot_dict['reference']
                    beam_x_ref = ref_dict['beam_xy'][:, 0]
                    beam_y_ref = ref_dict['beam_xy'][:, 1]
                    beam_z_ref = ref_dict['zs_of_interest']
                    self.ref_line_zx.set_ydata(beam_x_ref / x_unit)
                    self.ref_line_zx.set_xdata(beam_z_ref / z_unit)
                    self.ref_line_zy.set_ydata(beam_y_ref / y_unit)
                    self.ref_line_zy.set_xdata(beam_z_ref / z_unit)


        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        self.pipe = pipe
        self.fig, self.ax = plt.subplots(2, 1, figsize=(14, 8))

        # ZX
        cax = self.ax[0]
        cax.set_title('ZX')
        cax.grid()
        # cax.set_xlabel('z / m')  # Removed as it overlaps title of next plot.
        cax.set_ylabel('x / mm')
        self.line_zx, = cax.plot([0], [0], '-b.', label='current')
        self.points_zx, = cax.plot([0], [0], 'bd')
        self.ref_line_zx, = cax.plot([0], [0], '-kx', label='reference')
        cax.legend()

        # ZY
        cax = self.ax[1]
        cax.set_title('ZY')
        cax.grid()
        cax.set_xlabel('z / m')
        cax.set_ylabel('y / mm')
        self.line_zy, = cax.plot([0], [0], '-r.', label='current')
        self.points_zy, = cax.plot([0], [0], 'rd')
        self.ref_line_zy, = cax.plot([0], [0], '-kx', label='reference')
        cax.legend()

        self.ax[0].set_ylim([-10, 10])
        self.ax[0].set_xlim([-1, 110])
        self.ax[1].set_ylim([-10, 10])
        self.ax[1].set_xlim([-1, 110])

        # Draw reference

        timer = self.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.call_back)
        timer.start()
        plt.tight_layout()
        plt.show()


        self.pipe_grid = pipe
        self.nrow, self.ncol = (2, 2)
        self.fig_grid, self.ax_grid = plt.subplots(self.nrow, self.ncol)
        for i in range(self.nrow):
            for j in range(self.ncol):
                cax = self.ax_grid[i, j]
                self.image_plots.append(cax.imshow())
                print(type(self.image_plots[-1]))

        timer_grid = self.fig.canvas.new_timer(interval=100)
        timer_grid.add_callback(self.call_back)
        timer.start()
        plt.show()


class PltPlotter:
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
    pl = PltPlotter()
    for _ in range(10000):
        plot_dict = {
            'beam_xy': numpy.random.rand(4, 2),
            'beam_angles': numpy.random.rand(4, 2),
            'zs_of_interest': numpy.random.rand(4),
            'uv_points': numpy.random.rand(4, 2),
        }
        pl.plot(plot_dict)
        time.sleep(0.3)
    pl.plot(finished=True)


if __name__ == '__main__':
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")
    main()