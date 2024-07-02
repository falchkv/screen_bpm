import h5py
import numpy

from screen_bpm.lm_screen_analysis.image_operations import extract_beam_position
from screen_bpm.lm_screen_analysis.ray_operations import fit_ray, \
    compute_beam_position, compute_xy_angles
from screen_bpm.lm_screen_analysis.homography_screen import HomographyScreen


class ScreenBPM:
    def __init__(self, screens):
        self.screens = screens

    def compute_beam_metrics(self, uv_points, z_beam_position, sigmas=None):
        """
        Computes the position of intersection of rays with the plane
        z=z_beam_position, as well as the angle of the vertically and
        horizontally projected beam with respect to the z-axis, referred to as
        the x and y angels, respectively.

        Parameters
        ----------
        uv_points : numpy.ndarray, tuple or list
            Array of dimension (n_screens, 2).

        z_beam_position : float
            Z-coordinate of plane where beam intersection is to be computed.

        sigmas : numpy.ndarray, optional
            An estimate of the uncertainty of the provided xyz_positions. These
            values are used to weigh the data points in the fitting.

        Returns
        -------
        numpy.ndarray
            The x and y coordinates of the intersections.

        numpy.ndarray
            The x and y angles
        """
        if len(uv_points) != len(self.screens):
            raise ValueError(
                'Number of uv_points must be equal to number of screens.'
            )

        xyz_positions = []
        for screen, uv_point in zip(self.screens, uv_points):
            if uv_point is not None:
                xyz = screen.uv_to_xyz(uv_point)[0]
                xyz_positions.append(xyz)

        ray_nodes = fit_ray(xyz_positions, sigmas=sigmas)

        beam_position = compute_beam_position(ray_nodes, z_beam_position)
        xy_angles = compute_xy_angles(ray_nodes)

        return beam_position, xy_angles

    def beam_metrics_from_images(
            self, images, z_beam_position):
        """
        Computes the position of intersection of rays with the plane
        z=z_beam_position, as well as the angle of the vertically and
        horizontally projected beam with respect to the z-axis, referred to as
        the x and y angels, respectively.

        Parameters
        ----------
        images : numpy.ndarray, tuple or list
            The screen images of the beam.

        z_beam_position : float
            Z-coordinate of plane where beam intersection is to be computed.

        Returns
        -------
        numpy.ndarray
            The x and y coordinates of the intersections.

        numpy.ndarray
            The x and y angles
        """
        if len(images) != len(self.screens):
            raise ValueError(
                'Number of images must be equal to number of screens.'
            )

        uv_points = []
        sigmas = []
        for image in images:
            uv_point, sigma = extract_beam_position(image)
            uv_points.append(uv_point)
            sigmas.append(sigma)

        return self.compute_beam_metrics(
            uv_points, z_beam_position, sigmas
        )


def write_calibration(screen_bpm, path, screen_names=None):
    """
    Writes a screen bpm calibration to an h5 file.

    parameters
    ----------
    screen_bpm : ScreenBPM
        ScreenBPM object whose calibration is to be written to file.

    path : str
        The path to the file.

    screen_names : list, tuple, optional
        Name of screens. If None, screens will be named screen_0 ... screen_n-1.

    """
    #  Create screen names of None are given.
    if screen_names is None:
        screen_names = ['screen_{}'.format(i) for i in range(len(screen_bpm.screens))]

    with h5py.File(path, 'w') as h5:
        for screen, screen_name in zip(screen_bpm.screens, screen_names):
            h5.create_dataset(
                name='screens/{}/homography'.format(screen_name), data=numpy.array(screen.homography))
            h5.create_dataset(
                name='screens/{}/z_position'.format(screen_name), data=screen.z_position)


def load_calibration(path):
    """
    Reads a screen bpm calibration from an h5 file.

    parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    ScreenBPM
        A calibrated ScreenBPM object.

    list
        List of screen names.
    """
    screen_names = []
    screens = []
    with h5py.File(path, 'r') as h5:
        for screen_name in h5['screens']:
            homography = h5['screens/{}/homography'.format(screen_name)][()]
            z_position = h5['screens/{}/z_position'.format(screen_name)][()]
            screen = HomographyScreen(z_position, homography)
            screens.append(screen)
            screen_names.append(screen_name)

    return ScreenBPM(screens), screen_names
