import time
from glob import glob
import os
from dateutil import parser

import numpy

from screen_bpm.viewer.polling import TangoPoller

from screen_bpm.general_io import load_lm_screen_image, load_lm_screen_images


class LMScreenTangoTriggerer():
    def __init__(self, poll_targets, debug_mode=False, poll_rate=3.0):
        self.poller = TangoPoller(poll_targets=poll_targets)
        self.poll_rate = poll_rate  # Hz
        self.t_triggered = time.time() - 999
        self.latest_image_times = {poll_target: -999 for poll_target in poll_targets}  # time of new image in tango server
        self.debug_mode = debug_mode

        ### set initial trigger info
        if self.debug_mode:
            self._debug_poll()
        else:
            self.poll()

    def poll(self):
        """
        Polls tango server for images
        """
        poll_res = self.poller.poll_all()
        for key, previous_image_time in self.latest_image_times.items():
            time_string = poll_res[key]
            latest_image_time = parser.parse(time_string).timestamp()
            if latest_image_time > previous_image_time:
                self.latest_image_times[key] = latest_image_time
                self.set_trigger_info(latest_image_time, self.t_triggered)
                self.t_triggered = time.time()
                #print('triggered: {}'.format(latest_image_time))

    def _debug_poll(self):
        print('DEBUG POLLING')
        self.t_triggered = time.time()
        self.latest_image_time = 0
        self.set_trigger_info(
            self.latest_image_time, self.t_triggered)
        print('DEBUG POLLING DONE')

    def set_trigger_info(self, image_time, t_triggered):
        self.trigger_info = {
            'image_time': image_time,
            't_triggered': t_triggered
        }

    def get_trigger_info(self):
        """

        Returns
        -------
        dict
            The trigger info
        """
        return self.trigger_info

    def start_monitoring(self):
        """
        Starts monitoring.
        """
        poll_delay = 1.0 / self.poll_rate

        while True:
            time.sleep(poll_delay)
            if self.debug_mode:
                self._debug_poll()
            else:
                self.poll()  # triggered if there is a new lm screen image


def get_lm_dict(keys=('lm2', 'lm3', 'lm4')):
    if isinstance(keys, str):
        keys = (keys, )
    to_poll = {key: POLL_TAREGETS[key] for key in keys}
    poller = TangoPoller(poll_targets=to_poll)
    poll_res = poller.poll_all()
    return poll_res


class LMScreenDataLoader:
    def __init__(self, screen_names, debug_mode=False):
        self.screen_names = screen_names  # name of screens to load, e.g. 'LM2'
        self.debug_mode = debug_mode
        # to_poll = {key: POLL_TAREGETS[key] for key in screen_names}
        # self.poller = TangoPoller(poll_targets=to_poll)

    def load(self, trigger_info):  # input parameters to be re-considered
        """
        Loads the data from a tango server.

        Parameters
        ----------

        Returns
        -------
        dict
            The data
        """
        print('about to load...')
        lm_dict = get_lm_dict(keys=self.screen_names)  # function from macro testing?
        return lm_dict

POLL_TAREGETS = {
    'mscopezoom': ('haspp06mc01:10000/p06/mscope/mi.01', 'Zoom'),
    'microdiagy': ('haspp06mc01:10000/p06/motor/mi.33', 'Position'),
    'ps2hgap': ('haspp06mono:10000/p06/vmexecutors/ps2horizontalgap', 'Position'),
    'ps2hoff': ('haspp06mono:10000/p06/vmexecutors/ps2horizontaloffset', 'Position'),
    'lm2': ("haspp06:10000/p06/lmscreen/lm2", 'Frame'),
    'lm3': ("haspp06:10000/p06/lmscreen/lm3", 'Frame'),
    'lm4': ("haspp06:10000/p06/lmscreen/lm4", 'Frame'),
    'LM2': ("haspp06:10000/p06/lmscreen/lm2", 'Frame'),
    'LM3': ("haspp06:10000/p06/lmscreen/lm3", 'Frame'),
    'LM4': ("haspp06:10000/p06/lmscreen/lm4", 'Frame'),
    'xrayeye': ('haspp06mc01:10000/p06/motor/mi.48', 'Position'),
    'mscope_raw': ("haspp06:10000/p06/tangovimba/exp.01", 'ImageRaw'),
    'cctv1': ("haspp06:10000/p06/cctv/exp.01", 'ImageRaw'),
    'cctv2': ("haspp06:10000/p06/cctv/exp.02", 'ImageRaw'),
    'cctv3': ("haspp06:10000/p06/cctv/exp.03", 'ImageRaw'),
    'cctv4': ("haspp06:10000/p06/cctv/exp.04", 'ImageRaw'),
    'cctv5': ("haspp06:10000/p06/cctv/exp.05", 'ImageRaw'),
    'cctv6': ("haspp06:10000/p06/cctv/exp.06", 'ImageRaw'),
}
if __name__ == '__main__':

    def get_cctv_dict(keys=('cctv6')):
        if isinstance(keys, str):
            keys = (keys,)
        to_poll = {key: POLL_TAREGETS[key] for key in keys}
        poller = TangoPoller(poll_targets=to_poll)
        poll_res = poller.poll_all()
        return poll_res


    cctv_dict = get_cctv_dict(keys=('cctv6'))
    import h5py

    with h5py.File('tmp.h5', 'w') as h5:
        h5.create_dataset(data=cctv_dict['cctv6'], name='cctv6')
    from matplotlib import pyplot as plt
    plt.imshow(cctv_dict['cctv6'])
    plt.show()
    poll_targets_trigger = {
        'lm2': ("haspp06:10000/p06/lmscreen/lm2", 'FrameTimeStr'),
        'lm3': ("haspp06:10000/p06/lmscreen/lm3", 'FrameTimeStr'),
        'lm4': ("haspp06:10000/p06/lmscreen/lm4", 'FrameTimeStr'),
    }
    poll_targets_load = {
        'lm2': ("haspp06:10000/p06/lmscreen/lm2", 'Frame'),
        'lm3': ("haspp06:10000/p06/lmscreen/lm3", 'Frame'),
        'lm4': ("haspp06:10000/p06/lmscreen/lm4", 'Frame'),
    }

    #triggerer = LMScreenTangoTriggerer(poll_targets_trigger)
    #triggerer.start_monitoring()
    loader = LMScreenDataLoader(('LM3', 'LM4'))
    img = loader.load(None)
    while True:
        cctv_dict = get_cctv_dict(keys=('cctv5'))
        plt.imshow(cctv_dict['cctv5'])
        plt.show()
        time.sleep(1)
    print(img)