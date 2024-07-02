import pytest

from screen_bpm.lm_screen_analysis.ray_operations import *


@pytest.mark.parametrize(
    "ray_nodes, z_beam_position, expected",
    [
        ((0, 0, 0, 0, 0, 1), 10, (0, 0)),
        ((0, 0, 0, 1, 0, 1), 10, (10, 0)),
        ((0, 0, 0, 0, -1, 1), 10, (0, -10)),
        ((0, -1, 0, 0, -1, 1), 10, (0, -1)),
        ([0.5, -1, 0, 1, -1, 1], 0, (0.5, -1)),
        ([0.5, -1, 0, 1, -1, 1], [0, 1], ((0.5, -1), (1, -1))),
    ]
)
def test_compute_beam_position(ray_nodes, z_beam_position, expected):
    """
    Tests for compute_beam_position
    """
    beam_position = compute_beam_position(ray_nodes, z_beam_position)
    numpy.testing.assert_array_equal(beam_position, expected)


@pytest.mark.parametrize(
    "ray_nodes, expected",
    [
        ((0, 0, 0, 0, 0, 1), (0, 0)),
        ((0, 0, 0, 1, 0, 1), (numpy.pi/4, 0)),
        ((0, 0, 0, 0, -1, 1), (0, -numpy.pi/4)),
        ((0, -1, 0, 0, -1, 1), (0, 0)),
        ([0.5, -1, 0, 1, -1, 1], (numpy.arctan(0.5/1.0), 0)),
    ]
)
def test_compute_xy_angles(ray_nodes, expected):
    """
    Tests for compute_beam_position
    """
    xy_angles = compute_xy_angles(ray_nodes)
    numpy.testing.assert_array_equal(xy_angles, expected)


@pytest.mark.parametrize(
    "xyz_positions, sigmas, expected",
    [
        (
                [
                    [0, 0, 0],
                    [0, 0, 1]
                ],
                None,
                [0, 0, 0, 0, 0, 1]
        ),
        (
                [
                    [0, 0, 0],
                    [0, 0, 1]
                 ],
                (1, 1),
                [0, 0, 0, 0, 0, 1]
        ),
        (
                [
                    [0, 0, 0],
                    [99, -99, 0.5],
                    [0, 0, 1]
                ],
                (1, 1e10, 1),
                [0, 0, 0, 0, 0, 1]
        ),
        (
                [
                    [0, 0, 0],
                    [99, -99, 0.5],
                    [0, 0, 1]
                ],
                (1, 1, 1e10),
                [0, 0, 0, 198, -198, 1]
        ),
    ]
)
def test_fit_ray(xyz_positions, sigmas, expected):
    """
    Tests for compute_beam_position
    """
    ray_nodes = fit_ray(xyz_positions, sigmas=sigmas)
    numpy.testing.assert_array_almost_equal(ray_nodes, expected, decimal=5)
