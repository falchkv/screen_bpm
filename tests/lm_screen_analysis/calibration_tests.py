import pytest

from screen_bpm.lm_screen_analysis.calibration import *


@pytest.fixture
def ground_truth_calibrations():
    calibrations = {
        'lm2': {
            'screen_center': (1.0, 9.0),
            'position': (0, 0, 39.51),
            'euler_angles': (0, 0, 0),
            'pixel_size': (1.0e-5, 1.0e-5),
        },
        'lm3': {
            'screen_center': (2, 7),
            'position': (0, 0, 42.53),
            'euler_angles': (0, 0, 0),
            'pixel_size': (1.0e-5, 1.0e-5),
        },
        'lm4': {
            'screen_center': (1, 1),
            'position': (0, 0, 90),  # inaccurate
            'euler_angles': (0, 0, 0),
            'pixel_size': (1.0e-5, 1.0e-5),
        }
    }
    return calibrations


@pytest.fixture
def ground_truth_screens(ground_truth_calibrations):
    screens = calibrations_to_screens(ground_truth_calibrations)
    return screens


@pytest.fixture
def ground_truth_ray_nodes():
    ray_nodes = numpy.array([
        [0, 0, 0, 0.001, 0, 100],
        [0, 0, 0, 0, 0.001, 100],
        [0, 0, 0, 0.001, 0.001, 100],
        [0, 0, 0, 0.001, -0.001, 100],
    ])
    return ray_nodes


@pytest.fixture
def ground_truth_uv_intersections(
        ground_truth_ray_nodes,
        ground_truth_screens
):
    uv = evaluate_screen_positions(
        ground_truth_ray_nodes,
        ground_truth_screens
    )
    return uv


def test_calibration_to_array(ground_truth_calibrations):
    """
    tests for calibration_to_array and its inverse, array_to_calibration.
    """
    calibration_mask = {
        'lm2': {
            'screen_center': (True, True),
            'position': (False, False, True),
            'euler_angles': (True, False, False),
            'pixel_size': (False, False),
        },
        'lm3': {
            'screen_center': (True, False),
            'position': (False, True, False),
            'euler_angles': (True, False, True),
            'pixel_size': (False, True),
        },
        'lm4': {
            'screen_center': (False, True),
            'position': (True, True, True),
            'euler_angles': (False, False, False),
            'pixel_size': (False, True),
        }
    }

    calibration_array = calibration_to_array(
        ground_truth_calibrations, calibration_mask
    )

    inverse = array_to_calibration(
        calibration_array,
        calibration_mask,
        ground_truth_calibrations
    )

    for screen_name, calibration in ground_truth_calibrations.items():
        for key, value in calibration.items():
            numpy.testing.assert_array_equal(inverse[screen_name][key], value)


def test_calibration_scaling(ground_truth_calibrations):
    """
    tests for calibration_to_array and its inverse, array_to_calibration.
    """
    calibration_mask = {
        'lm2': {
            'screen_center': (True, True),
            'position': (False, False, True),
            'euler_angles': (True, False, False),
            'pixel_size': (False, False),
        },
        'lm3': {
            'screen_center': (False, False),
            'position': (False, False, False),
            'euler_angles': (False, False, True),
            'pixel_size': (False, False),
        },
        'lm4': {
            'screen_center': (False, False),
            'position': (False, False, False),
            'euler_angles': (False, False, False),
            'pixel_size': (False, False),
        }
    }
    # Initialize calibration scaler
    scaler = default_calibration_scaler()

    # Create calibration_array
    calibration_array = calibration_to_array(
        ground_truth_calibrations, calibration_mask
    )
    # Create scaled calibration_array
    calibration_array_scaled = apply_calibration_scaler(
        calibration_array, scaler, calibration_mask
    )

    # Create unscaled calibration_array
    calibration_array_unscaled = unapply_calibration_scaler(
        calibration_array_scaled, scaler, calibration_mask
    )

    # assert correct inverse
    numpy.testing.assert_array_equal(
        calibration_array_unscaled, calibration_array
    )


def test_compute_error(
        ground_truth_ray_nodes,
        ground_truth_screens,
        ground_truth_uv_intersections
):
    """
    Tests that initial error is 0.
    """
    initial_error = compute_error(
        ground_truth_uv_intersections,
        ground_truth_ray_nodes,
        ground_truth_screens,
    )

    assert initial_error == 0
