import os
import glob
import tempfile

import pytest
import numpy
from p06io.images.image import Image

from screen_bpm.lm_screen_analysis.screen_bpm import ScreenBPM, write_calibration, load_calibration
from screen_bpm.lm_screen_analysis.screen import Screen
from screen_bpm.lm_screen_analysis.homography_screen import HomographyScreen

TEST_DATA_PATH = os.path.join('tests', 'test_data', 'lm_screen_analysis')


@pytest.mark.parametrize(
    "uv_points, expected_position, expected_xy_angles",
    [
        (
            [[0, 0], [0, 0], [0, 0]],
            (0, 0),
            (0, 0)
        ),
        (
                [[0, 1], [0, 1], [0, 1]],
                (-1, 0),
                (0, 0)
        ),
    ]
)
def test_compute_screen_metrics(
        uv_points, expected_position, expected_xy_angles
):
    """
    Tests for compute_screen_metrics
    """
    # Initialize screens
    euler_angles = [0, 0, 0]
    pixel_size = (1, 1)
    screen_center = [0, 0]
    pixel_size = 1  # not relevant
    screens = [
        Screen([0, 0, 0], euler_angles, pixel_size, screen_center),
        Screen([0, 0, 0.5], euler_angles, pixel_size, screen_center),
        Screen([0, 0, 1], euler_angles, pixel_size, screen_center),
    ]

    # Create screen BPM object
    screen_bpm = ScreenBPM(screens)

    beam_position, xy_angles = screen_bpm.compute_beam_metrics(uv_points, 1.0)

    numpy.testing.assert_array_almost_equal(beam_position, expected_position)
    numpy.testing.assert_array_almost_equal(xy_angles, expected_xy_angles)

@pytest.mark.parametrize(
    "uv_points, expected_position, expected_xy_angles",
    [
        (
            [[0, 0], [0, 0], [0, 0]],
            (0, 0),
            (0, 0)
        ),
        (
            [[0, 1], [0, 1], [0, 1]],
            (0, 1),
            (0, 0)
        ),
        (
            [[1, 0], [0, 0], [-1, 0]],
            (0, 0),
            (numpy.arctan(-1), 0)
        ),
        (
            [[1, -1], [1, 0], [1, 1]],
            (1, 0),
            (0, numpy.arctan(1))
        ),
    ]
)
def test_compute_screen_metrics_homography(
        uv_points, expected_position, expected_xy_angles
):
    """
    Tests for compute_screen_metrics using homography screen
    """
    # Initialize screens
    z1 = 0.0
    z2 = 1.0
    z3 = 2.0
    h1 = numpy.eye(3)
    h2 = numpy.eye(3)
    h3 = numpy.eye(3)

    screens = [
        HomographyScreen(z1, h1),
        HomographyScreen(z2, h2),
        HomographyScreen(z3, h3),
    ]

    # Create screen BPM object
    screen_bpm = ScreenBPM(screens)

    beam_position, xy_angles = screen_bpm.compute_beam_metrics(uv_points, 1.0)

    numpy.testing.assert_array_almost_equal(beam_position, expected_position)
    numpy.testing.assert_array_almost_equal(xy_angles, expected_xy_angles)

@pytest.mark.parametrize(
    "directory_path, screen_centers, screen_positions, expected",
    [
        (
            os.path.join(TEST_DATA_PATH, '12000_ev'),
            (
                (162, 298),
                (216, 219),
                (319, 298)
            ),
            (
                (0, 0, 0),
                (0, 0, 10),
                (0, 0, 50)
            ),
            (0, 0)),
    ]
)
def test_beam_metrics_from_images(
        directory_path, screen_centers, screen_positions, expected
):
    """
    Tests for extract_beam_position
    """

    # Load images
    lm2_path = glob.glob(os.path.join(directory_path, 'lm2*'))[0]
    lm3_path = glob.glob(os.path.join(directory_path, 'lm3*'))[0]
    lm4_path = glob.glob(os.path.join(directory_path, 'lm4*'))[0]
    images = [
        Image().read(lm2_path),
        Image().read(lm3_path),
        Image().read(lm4_path),
    ]

    # Initialize screens
    euler_angles = [0, 0, 0]
    pixel_size = (1, 1)
    screen_center = [0, 0]
    pixel_size = 1  # not relevant
    screens = [
        Screen(
            screen_positions[0], euler_angles, pixel_size, screen_centers[0]
        ),
        Screen(
            screen_positions[1], euler_angles, pixel_size, screen_centers[1]
        ),
        Screen(
            screen_positions[2], euler_angles, pixel_size, screen_centers[2]
        ),
    ]

    # Create screen BPM object
    screen_bpm = ScreenBPM(screens)

    beam_position, xy_angles = screen_bpm.beam_metrics_from_images(
        images, 1.0
    )

    print(beam_position, xy_angles)

    numpy.testing.assert_array_almost_equal(beam_position, expected)


def test_write_read_calibration():
    """
    Tests for compute_screen_metrics using homography screen
    """
    path = os.path.join(tempfile.tempdir, 'screen_bpm_calibration.h5')
    screen_names = ['a', 'b', 'c']
    screens = []
    for i in range(len(screen_names)):
        homography = numpy.random.rand(3, 3)
        z_position = numpy.random.rand()
        screens.append(HomographyScreen(z_position, homography))

    bpm = ScreenBPM(screens)
    write_calibration(bpm, path, screen_names=screen_names)
    loaded_bpm, loaded_screen_names = load_calibration(path)

    assert len(loaded_screen_names) == len(screen_names)
    for i, screen_name in enumerate(screen_names):
        numpy.testing.assert_array_equal(loaded_bpm.screens[i].homography, bpm.screens[i].homography)
        numpy.testing.assert_array_equal(loaded_bpm.screens[i].z_position, bpm.screens[i].z_position)
        assert loaded_screen_names[i] == screen_name

    os.remove(path)