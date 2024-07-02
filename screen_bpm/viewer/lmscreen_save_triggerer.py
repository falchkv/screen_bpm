import time
from glob import glob
import os

import numpy

from screen_bpm.general_io import load_lm_screen_image, load_lm_screen_images


def debug_get_path():
    """
    Get random file path from test_data

    Returns
    -------
    str
        Path to datafile
    """
    file_list = glob(os.path.join(
        'tests', 'test_data',
        'lm_screen_counts', '*',
        'lm_screens', '*.h*5'
    ))
    # get random path
    path = file_list[int(numpy.random.rand() * len(file_list))]
    return path

class LMScreenSaveTriggerer():
    def __init__(self, experiment_path, debug_mode=False, poll_rate=3.0):
        if experiment_path is None:
            raise IOError(f'experiment_path is None')
        elif os.path.exists(experiment_path):
            self.experiment_path = experiment_path
        else:
            raise IOError(f'Could not access {experiment_path}')
        self.poll_rate = poll_rate  # Hz
        self.t_triggered = time.time() - 999

        self.latest_save_id = 0
        self.debug_mode = debug_mode
        ### set initial trigger info

        if self.debug_mode:
            self._debug_poll()
        else:
            self.poll()

    def poll(self):
        """
        Looks in experiment data and find the latest count with lm screens.
        Triggering occurs if the latest count id is larger than the previous largest count id.
        """
        search_string = os.path.join(
            self.experiment_path, 'raw', '*', 'count_*', 'lm_screens'
        )
        dir_list = sorted(glob(search_string), key=self.count_id_from_path)

        latest_save_id = self.count_id_from_path(dir_list[-1])
        latest_file = os.path.join(
            dir_list[-1], 'count_%.5i.hdf5' % latest_save_id)

        if latest_save_id > self.latest_save_id:
            self.latest_save_id = latest_save_id
            self.set_trigger_info(
                latest_file, latest_save_id, self.t_triggered)
            self.t_triggered = time.time()
            print('triggered: {}'.format(latest_save_id))

    def _debug_poll(self):
        print('DEBUG POLLING')
        self.t_triggered = time.time()
        self.latest_save_id = 0
        latest_file = debug_get_path()
        self.set_trigger_info(
            latest_file, self.latest_save_id, self.t_triggered)
        print('DEBUG POLLING DONE')

    @staticmethod
    def count_id_from_path(path):
        return int(path.split('count_')[-1][:5])

    def set_trigger_info(self, latest_file, count_id, t_triggered):
        print(latest_file, count_id, t_triggered)
        self.trigger_info = {
            'path': latest_file,
            'count_id': count_id,
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
                self.poll()  # triggered if new lm screen save is found


class LMScreenDataLoader:
    def __init__(self, screen_names, debug_mode=False):
        self.screen_names = screen_names  # name of screens to load, e.g. 'LM2'
        self.debug_mode = debug_mode

    def load(self, trigger_info):  # input parameters to be re-considered
        """
        Loads the data.

        Parameters
        ----------

        Returns
        -------
        dict
            The data
        """
        path = trigger_info['path']

        while True:
            try:
                time.sleep(1)
                data = load_lm_screen_images(path, self.screen_names)
                break
            except Exception as e:
                print(trigger_info)
                print('could not open {}. retrying'.format(path))

        return data


if __name__ == '__main__':
    from AutoKB.io.paths import experiment_path_from_id
    experiment_id = '11018189'
    experiment_path = experiment_path_from_id(experiment_id)
    triggerer = LMScreenSaveTriggerer(experiment_path)

    triggerer.start_monitoring()
