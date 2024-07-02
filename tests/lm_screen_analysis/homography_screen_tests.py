import pytest
import numpy

from screen_bpm.lm_screen_analysis.homography_screen import *


@pytest.mark.parametrize(
    "uv_point, z_position, xy_offset",
    [
        ((0, 0), 0.1, (0, 0)),
        ((0, 1), 100.0, (0, 0)),
        ((0, -1), 100, (0, 0)),
        ((1, 0), 10.0, (0, 0)),
        ((-1, 0), 10, (0, 0)),
        ((-1, 1), -100, (0, 0)),
        ((0, 0), 0.1, (1, 0)),
        ((0, 1), 100.0, (0, 1)),
        ((0, -1), 100, (-1, 0)),
        ((1, 0), 10.0, (0, -1)),
        ((-1, 0), 10, (1, 1)),
        ((-1, 1), -100, (-1, -1)),
    ]
)
def test_uv_to_xyz(uv_point, z_position, xy_offset):
    """
    Tests for uv to xyz conversion.
    """
    xy_offset = numpy.array(xy_offset)
    homography = numpy.eye(3)
    homography[0, 0] = 2  # x twice as large as u
    homography[1, 1] = 1.0/2  # y half as large as v
    expected_xyz = numpy.hstack([uv_point * numpy.diag(homography)[:2], z_position])
    expected_xyz[:2] += xy_offset

    screen = HomographyScreen(z_position, homography, xy_offset=xy_offset)

    # Compute and test xyz result
    xyz = screen.uv_to_xyz(uv_point)
    numpy.testing.assert_array_almost_equal(numpy.squeeze(xyz), expected_xyz, decimal=3)

    # Compute and test inverse result
    inverse_result = screen.xyz_to_uv(xyz)
    numpy.testing.assert_array_almost_equal(numpy.squeeze(inverse_result), uv_point, decimal=3)

    # Assert that z is always screen.z_position
    numpy.testing.assert_array_almost_equal(numpy.squeeze(xyz[:, -1]), screen.z_position, decimal=3)


@pytest.mark.parametrize(
    "ray_nodes, expected",
    [
        ((1, 0, 0, 0, 1, 1),  (1, 0, 0)),
        ((1, 0, 1, 0, 1, 0), (0, 1, 0)),
        ((1, 0, 10, 1, 0, -10), (1, 0, 0)),
        ((1, 1, 10, -1, -1, -10), (0, 0, 0)),

        (
            (
                (1, 0, 0, 0, 1, 1),
                (1, 0, 1, 0, 1, 0),
                (1, 0, 10, 1, 0, -10),
                (1, 0, -10, 1, 0, 10),
            ),
            (
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
            ),
        )  # multiple rays
    ]
)
def test_compute_ray_intersection(ray_nodes, expected):
    """
    Tests for compute_ray_intersection
    """
    homography = numpy.eye(3)
    homography[0, 0] = 2  # x twice as large as u
    homography[1, 1] = 1.0/2  # y half as large as v
    z_position = 0

    screen = HomographyScreen(z_position, homography)

    # Compute and test intersection
    intersection = screen.compute_ray_intersection(ray_nodes)

    if numpy.array(expected).size == 3:
        numpy.testing.assert_array_almost_equal(intersection[0], expected, decimal=3)
    else:
        numpy.testing.assert_array_almost_equal(intersection, expected, decimal=3)
