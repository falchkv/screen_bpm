import pytest
import numpy
from scipy.spatial.transform import Rotation

from screen_bpm.lm_screen_analysis.homography_calibration import *
from screen_bpm.lm_screen_analysis import camera_operations


def test_findHomography():
    """
    Tests for find_homogrpahy
    """
    n_points = 10
    # xy_in = (numpy.random.rand(2, n_points) - 0.5)
    # h_ground_truth = numpy.random.rand(3, 3)
    # Test may sometimes fail with random inputs due to numerical precicion. This is not what this test is assessing,
    # therefore, fixed inputs are used
    xy_in = numpy.array([
        [0.117485, -0.30992, 0.477846, 0.185303, -0.270749, 0.23332, 0.192931, 0.131668, -0.482793, -0.161537],
        [0.021106, -0.15953, -0.054297,  0.075567,  0.400321, -0.040694, -0.33599, 0.124789, -0.303484, -0.321174]
    ])
    h_ground_truth = numpy.array([
        [0.74263419, 0.30031479, 0.47816771],
        [0.95133956, 0.01441223, 0.43822347],
        [0.1614151, 0.7357518, 0.0061845]
    ])
    xy_in_hom = numpy.vstack([xy_in, numpy.ones((n_points, ))])

    xy_out_hom = h_ground_truth @ xy_in_hom
    xy_out = xy_out_hom[:2, :] / xy_out_hom[-1, :]
    h = find_homography(xy_in, xy_out_hom)
    reprojected_hom = h @ xy_in_hom
    reprojected = reprojected_hom[:2, :] / reprojected_hom[-1, :]

    numpy.testing.assert_array_almost_equal(reprojected, xy_out, decimal=1)


def test_screen_calibration():
    """
    Tests for calibration procedure, mimicing real calibration process.
    """
    z_screen = 50.0

    # Simulation parameters
    angles_y = numpy.linspace(-10e-6, 10e-6, 10)
    angles_x = numpy.linspace(-5e-6, 5e-6, 10)
    angles_X, angles_Y = numpy.meshgrid(angles_x, angles_y)
    projection_matrix = numpy.zeros((3, 4))
    projection_matrix[0, 0] = 1.0
    projection_matrix[1, 1] = 1
    projection_matrix[2, 3] = 1
    rotation_matrix = Rotation.from_euler(
            'XYZ', [0, 0, 0], degrees=False
        ).as_matrix()
    camera_matrix = numpy.diag([10, 10, 1])
    projection_matrix = camera_operations.synthesize_projection_matrix(
        camera_matrix,
        numpy.eye(3),
        (0, 0, 0, 1)
    )
    projection_matrix = camera_operations.rotate_projection_plane(projection_matrix, rotation_matrix)

    # Simulated data
    xy = numpy.tan(numpy.vstack([angles_X.flatten(), angles_Y.flatten()])) * z_screen
    xyz_hom = numpy.vstack([xy,  z_screen*numpy.ones(xy.shape[1]), numpy.ones(xy.shape[1])])
    uv_hom = projection_matrix @ xyz_hom
    uv = uv_hom[:2, :] / uv_hom[-1, :]

    h = find_homography(uv, xy)

    reprojected_hom = h @ uv_hom
    reprojected = reprojected_hom[:2, :] / reprojected_hom[-1, :]

    # Test
    numpy.testing.assert_array_almost_equal(reprojected, xy, decimal=10)

