import copy
import threading
import time

import numpy

#from lmscreen_save_triggerer import LMScreenSaveTriggerer, LMScreenDataLoader
from lmscreen_tango_triggerer import LMScreenTangoTriggerer, LMScreenDataLoader
from screen_bpm.lm_screen_analysis import screen_bpm
from screen_bpm.lm_screen_analysis.image_operations import (
    extract_max_position)
from screen_bpm.viewer.matplotlib_plotter import PltPlotter
from screen_bpm.viewer.image_grid_plotter import PltGridPlotter


class Viewer:
    def __init__(self, experiment_path, screen_names, screen_bpm,
                 xy_offsets={}, debug_mode=False, reference_uvs=None,
                 reference_screen_bpm=None, reference_screen_names=None,
                 update_triggerer=None):
        self.maximum_update_rate = 3.0  # Hz
        self.screen_names = screen_names
        self.xy_offsets = xy_offsets
        # self.debug_mode = debug_mode
        self.reference_uvs = reference_uvs
        self.plotter = PltPlotter(
            reference_uvs=reference_uvs, interval=self.maximum_update_rate)
        self.grid_plotter = PltGridPlotter(
            reference_uvs=reference_uvs, interval=self.maximum_update_rate)
        self.data_loader = LMScreenDataLoader(
            screen_names, debug_mode=debug_mode)


        if update_triggerer is not None:
            self.update_triggerer = update_triggerer
        else:
            self.update_triggerer = LMScreenSaveTriggerer(
               experiment_path, debug_mode=debug_mode,
               poll_rate=self.maximum_update_rate)
        self.frame_counter = 0

        self.screen_bpm = screen_bpm
        self.reference_screen_bpm = reference_screen_bpm
        self.reference_screen_names = reference_screen_names

    def start_monitoring(self):
        poll_delay = 1.0 / self.maximum_update_rate
        previous_t_triggered = time.time() - 999  # initialize
        trigger_thread = threading.Thread(
            target=self.update_triggerer.start_monitoring)
        trigger_thread.start()
        while True:
            if self.update_triggerer.t_triggered > previous_t_triggered:
                previous_t_triggered = self.update_triggerer.t_triggered
                trigger_info = self.update_triggerer.get_trigger_info()
                self.update_plot(trigger_info)
            time.sleep(poll_delay)

    def update_plot(self, trigger_info):
        print(trigger_info)
        images = self.data_loader.load(trigger_info)
        processed = self.process_data(images)
        plot_dict = processed.copy()
        plot_dict.update(trigger_info)
        grid_plot_dict = {
            #'count_id': plot_dict['count_id'],
            'images': images,
            'uv_points': processed['uv_points']
        }
        self.plotter.plot(plot_dict)
        self.grid_plotter.plot(grid_plot_dict)
        self.frame_counter += 1

    def process_data(self, data):

        self.xy_offsets['LM3'] = 10e-3*numpy.random.rand(2)# FAKE oFFSET
        # update xy_offsets
        for screen, screen_name in zip(self.screen_bpm.screens, self.screen_names):
            if screen_name in self.xy_offsets:
                screen.xy_offset = self.xy_offsets[screen_name]
        uv_points = Viewer.get_uv_points(data)

        # where should the beam position be evaluated
        screen_zs = [screen.z_position for screen in self.screen_bpm.screens]
        extra_zs = [0, 90, 95]
        zs = numpy.array(sorted(screen_zs + extra_zs))
        beam_xy, beam_angles = self.compute_beam(uv_points, zs)

        # Calculate screen xyz intersects
        ordered_uv_points = Viewer._order_uv_points(
            uv_points, self.data_loader.screen_names)
        screen_xyz = numpy.full(
            (3, len(ordered_uv_points)), fill_value=numpy.nan)
        for i, screen in enumerate(self.screen_bpm.screens):
            screen_xyz[:, i] = screen.uv_to_xyz(ordered_uv_points[i])

        processed_dict = {
            'beam_xy': beam_xy,
            'beam_angles': beam_angles,
            'zs_of_interest': zs,
            'uv_points': uv_points,
            'screen_xyz': screen_xyz,
        }

        # create reference beam
        if self.reference_uvs is not None and self.reference_screen_bpm is not None and self.reference_screen_names is not None:
            beam_xy_ref, beam_angles_ref = self.compute_reference_beam(zs)
            print(beam_xy_ref)
            processed_dict['reference'] = {
                'beam_xy': beam_xy_ref,
                'beam_angles': beam_angles_ref,
                'zs_of_interest': zs,
                'uv_points_ref': self.reference_uvs,
            }

        return processed_dict

    @staticmethod
    def get_uv_points(data):
        uv_points = {}
        for screen_name, image in data.items():
            u, v = extract_max_position(image)
            uv_points[screen_name] = numpy.hstack([u, v])
        return uv_points


    @staticmethod
    def _order_uv_points(uv_points, screen_names):
        """
        Screen bpm needs to be fed uv points in correct order. This method
        orders a dictionary of uv points accoding to correctly ordered keys.
        Parameters
        ----------
        uv_points : dict
            Dictionary of uv points, whose keys are the keys are screen names.

        screen_names : list or tuple
            Correctly ordered sequence of screen names.

        Returns
        -------
        numpy.ndarray
            Ordered uv points.
        """
        ordered_uv_points = [None] * len(screen_names)
        for screen_name, uv_point in uv_points.items():
            if screen_name in screen_names:
                index = numpy.where(
                    numpy.array(screen_names) == screen_name
                )[0][0]
                ordered_uv_points[index] = uv_point
        return ordered_uv_points


    def compute_beam(self, uv_points, z_beam_positions):
        """
        Parameters
        ----------

        uv_points : dict
            Dictionary of uv points, whose keys are the keys are screen names.

        z_beam_positions : numpy.ndarray
            Z-coordinates of plane where beam intersection is to be computed.

        Returns
        -------
        numpy.ndarray
            The x and y coordinates of the intersections.

        numpy.ndarray
            The x and y angles
        """
        # self.screen_bpm requires uv points to be fed in correct order
        ordered_uv_points = Viewer._order_uv_points(uv_points, self.screen_names)
        beam_xy, beam_angles = self.screen_bpm.compute_beam_metrics(
            ordered_uv_points, z_beam_positions
        )
        return beam_xy, beam_angles

    def compute_reference_beam(self, z_beam_positions):
        """
        Parameters
        ----------
        z_beam_positions : numpy.ndarray
            Z-coordinates of plane where beam intersection is to be computed.

        Returns
        -------
        numpy.ndarray
            The x and y coordinates of the intersections.

        numpy.ndarray
            The x and y angles
        """
        # self.screen_bpm requires uv points to be fed in correct order
        ordered_uv_points = Viewer._order_uv_points(self.reference_uvs, self.reference_screen_names)
        beam_xy, beam_angles = self.reference_screen_bpm.compute_beam_metrics(
            ordered_uv_points, z_beam_positions
        )
        return beam_xy, beam_angles


if __name__ == '__main__':
    from screen_bpm.io.paths import experiment_path_from_id
    experiment_id = '11018189'
    experiment_id = '11018129'
    experiment_path = experiment_path_from_id(experiment_id)

    calibration_path = 'lm_screen_calibration.h5'
    bpm, screen_names = screen_bpm.load_calibration(calibration_path)

    ref_uv = {
        'LM2': (230, 434),
        'LM3': (242, 220),
        'LM4': (340, 269),
    }
    xy_offsets = {
        'LM2': numpy.zeros((2, )),
        'LM3': numpy.zeros((2, )),
        'LM4': numpy.zeros((2, )),
    }
    # remove LM2, as it is not very well calibrated:
    index = [i for i, screen_name in enumerate(screen_names) if screen_name=='LM2'][0]
    screen_names.pop(index)
    bpm.screens.pop(index)

    poll_targets_trigger = {
        'lm2': ("haspp06:10000/p06/lmscreen/lm2", 'FrameTimeStr'),
        'lm3': ("haspp06:10000/p06/lmscreen/lm3", 'FrameTimeStr'),
        'lm4': ("haspp06:10000/p06/lmscreen/lm4", 'FrameTimeStr'),
    }
    triggerer = LMScreenTangoTriggerer(poll_targets_trigger)
    viewer = Viewer(
        experiment_path, screen_names, bpm, debug_mode=False,
        reference_uvs=ref_uv, reference_screen_bpm=copy.deepcopy(bpm), reference_screen_names=screen_names, update_triggerer=triggerer, xy_offsets=xy_offsets
    )
    viewer.start_monitoring()

