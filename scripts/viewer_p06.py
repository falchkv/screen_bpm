import copy
import threading
import os
import time

import numpy


from screen_bpm.viewer.lmscreen_tango_triggerer import LMScreenTangoTriggerer
from screen_bpm.lm_screen_analysis import screen_bpm
from screen_bpm.viewer.polling import TangoPoller
from screen_bpm.viewer.viewer import Viewer

if __name__ == '__main__':
    calibration_path = os.path.join('tests', 'test_data', 'calibration_1.h5')
    bpm, screen_names = screen_bpm.load_calibration(calibration_path)

    ref_uv = {
        'LM2': (230, 434),
        'LM3': (242, 220),
        'LM4': (340, 269),
    }
    xy_offsets = {
        'LM2': numpy.zeros((2, )),
        'LM3': numpy.array([0, 0]),
        'LM4': numpy.array([0, 0]),
    }
    # remove LM2, as it is not very well calibrated due to being too close to
    # the monochromator:
    index = [
        i for i, screen_name in enumerate(screen_names) if screen_name=='LM2'
    ][0]
    screen_names.pop(index)
    bpm.screens.pop(index)

    # where should the beam position be evaluated
    screen_zs = [screen.z_position for screen in bpm.screens]
    extra_zs = [0, 90, 95]
    zs = numpy.array(sorted(screen_zs + extra_zs))
    z_labels = ['source', 'LM3', 'LM4', 'micro', 'ptynami']

    poll_targets_trigger = {
        'lm2': ("haspp06:10000/p06/lmscreen/lm2", 'FrameTimeStr'),
        'lm3': ("haspp06:10000/p06/lmscreen/lm3", 'FrameTimeStr'),
        'lm4': ("haspp06:10000/p06/lmscreen/lm4", 'FrameTimeStr'),
    }

    def update_offset():
        """
        Returns a dictionary of xy offsets for a set of screen names.
        """
        # The solution of passing this function to the viewer object is
        # intended to keep overly specific syntax out of the viewer class.
        poll_targets = {
            'microdiagy': ("haspp06mc01:10000/p06/motor/mi.33", 'Position'),
            'diagy': ("haspp06mc01:10000/p06/motor/mi.07", 'Position'),
        }
        poller = TangoPoller(poll_targets=poll_targets)
        poll_res = poller.poll_all()
        xy_offsets = {
            'LM2': numpy.array([0e-3, 0]),
            'LM3': numpy.array([0, 0]),
            'LM4': numpy.array([poll_res['microdiagy'] - -4.9726, 0]) * 1e-3,
        }
        return xy_offsets

    triggerer = LMScreenTangoTriggerer(poll_targets_trigger)
    viewer = Viewer(
        None, screen_names, bpm, zs, debug_mode=False,
        reference_uvs=ref_uv, reference_screen_bpm=copy.deepcopy(bpm),
        reference_screen_names=screen_names, update_triggerer=triggerer,
        xy_offsets=xy_offsets, update_offset_func=update_offset,
        z_labels=z_labels
    )
    viewer.start_monitoring()

