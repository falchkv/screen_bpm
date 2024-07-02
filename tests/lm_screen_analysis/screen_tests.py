import pytest
import numpy

from screen_bpm.lm_screen_analysis.screen import Screen


@pytest.mark.parametrize(
    "pixel_size, screen_center, uv_point, uv_prime_point",
    [
        ((10, 500), (6, 7), (6, 7), (0, 0)),
        ((1, 5), (0, 0), (6, 5), (6, 5*5)),
    ]
)
def test__uv_to_uv_prime(pixel_size, screen_center, uv_point, uv_prime_point):
    """
    Tests for _uv_to_uv_prime conversion
    """
    euler_angles = [0, 0, 0]  # not relevant
    position = 0  # not relevant

    screen = Screen(position, euler_angles, pixel_size, screen_center)

    # Compute and test uv_prime
    uv_prime = screen._uv_to_uv_prime(uv_point)
    numpy.testing.assert_array_equal(uv_prime[0], uv_prime_point)

    # Compute and test inverse result
    inverse_result = screen._uv_prime_to_uv(uv_prime)
    numpy.testing.assert_array_equal(inverse_result[0], uv_point)


@pytest.mark.parametrize(
    "uv_prime_point, position, euler_angles, xyz_point",
    [
        ((0, 0), (123, 456, 789), (11, 22, 33), (123, 456, 789)),
        ((1, 0), (0, 0, 0), (0, 0, 0), (0, -1, 0)),
        ((0, 1), (0, 0, 0), (0, 0, 0), (-1, 0, 0)),

        ((1, 0), (0, 0, 0), (numpy.pi/2, 0, 0), (1, 0, 0)),  # roll
        ((0, 1), (0, 0, 0), (numpy.pi/2, 0, 0), (0, -1, 0)),  # roll
        ((1, 0), (0, 0, 0), (0, numpy.pi/2, 0), (0, 0, -1)),  # pitch
        ((0, 1), (0, 0, 0), (0, numpy.pi/2, 0), (-1, 0, 0)),  # pitch
        ((1, 0), (0, 0, 0), (0, 0, numpy.pi/2), (0, -1, 0)),  # yaw
        ((0, 1), (0, 0, 0), (0, 0, numpy.pi/2), (0, 0, 1)),  # yaw

        # roll then pitch
        ((1, 0), (0, 0, 0), (numpy.pi/2, numpy.pi/2, 0), (0, 0, -1)),

        # roll then yaw
        ((1, 0), (0, 0, 0), (numpy.pi/2, 0, numpy.pi/2), (1, 0, 0)),

        # pitch then yaw
        ((1, 0), (0, 0, 0), (0, numpy.pi/2, numpy.pi/2), (0, 0, -1)),

        # roll then pitch then yaw
        ((1, 0), (0, 0, 0), (numpy.pi/2, numpy.pi/2, numpy.pi/2), (0, 0, -1)),
    ]
)
def test__uv_prime_to_xyz(uv_prime_point, position, euler_angles, xyz_point):
    """
    Tests for _xyz_to_uv_prime, and _uv_prime_to_xyz conversion
    """
    screen_center = [0, 0]  # not relevant
    pixel_size = 1 # not relevant

    screen = Screen(position, euler_angles, pixel_size, screen_center)

    # Compute and test xyz result
    xyz = screen._uv_prime_to_xyz(uv_prime_point)
    numpy.testing.assert_array_almost_equal(xyz[0], xyz_point, decimal=3)

    # Compute and test inverse result
    inverse_result = screen._xyz_to_uv_prime(xyz)
    numpy.testing.assert_array_almost_equal(inverse_result[0], uv_prime_point, decimal=3)


@pytest.mark.parametrize(
    "euler_angles, position, expected",
    [
        ((numpy.pi/2, 0, 0),  (0, 0, 0), (0, 0, 1)),  # roll
        ((0, numpy.pi/2, 0),  (0, 0, 0), (0, -1, 0)),  # pitch
        ((0, 0, numpy.pi/2),  (0, 0, 0), (1, 0, 0)),  # yaw

        ((0, numpy.pi / 4, 0), (1, 0, 0), (0, -1/numpy.sqrt(2), 1/numpy.sqrt(2))),  # pitch
        ((0, 0, numpy.pi / 4), (1, 0, 0), (1/numpy.sqrt(2), 0, 1/numpy.sqrt(2))),  # yaw
    ]
)
def test_compute_normal(euler_angles, position, expected):
    """
    Tests for compute_normal.
    """
    screen_center = (0, 0)  # not relevant
    pixel_size = 1  # not relevant

    screen = Screen(position, euler_angles, pixel_size, screen_center)

    # Compute and test normal
    normal = screen.compute_normal()
    numpy.testing.assert_array_almost_equal(normal, expected, decimal=3)


@pytest.mark.parametrize(
    "euler_angles, position, ray_nodes, expected",
    [
        ((numpy.pi/4, 0, 0), (0, 0, 0), (1, 0, 0, 1, 0, 1),  (1, 0, 0)),  # roll
        ((0, numpy.pi/4, 0), (0, 0, 0), (1, 0, 0, 1, 0, 1), (1, 0, 0)),  # pitch
        ((0, 0, numpy.pi/4), (0, 0, 0), (1, 0, 0, 1, 0, 1), (1, 0, -1)),  # yaw

        ((numpy.pi / 4, 0, 0), (1, 0, 0), (1, 0, 0, 1, 0, 1), (1, 0, 0)),  # roll
        ((0, numpy.pi / 4, 0), (1, 0, 0), (1, 0, 0, 1, 0, 1), (1, 0, 0)),  # pitch
        ((0, 0, numpy.pi / 4), (1, 0, 0), (1, 0, 0, 1, 0, 1), (1, 0, 0)),  # yaw

        ((numpy.pi / 4, 0, 0), (0, 1, 0), (1, 0, 0, 1, 0, 1), (1, 0, 0)),  # roll
        ((0, numpy.pi / 4, 0), (0, 1, 0), (1, 0, 0, 1, 0, 1), (1, 0, -1)),  # pitch
        ((0, 0, numpy.pi / 4), (0, 1, 0), (1, 0, 0, 1, 0, 1), (1, 0, -1)),  # yaw

        (
            (0, 0, 0),
            (0, 0, 0),
            (
                (-1, 0, 1, 1, 0, 0),
                (0, 0, 1, 0, 0, -1),
                (0, 1, 1, 0, 1, -2),
                (2, 0, 1, 2, 0, -3),
            ),
            (
                (1, 0, 0),
                (0, 0, 0),
                (0, 1, 0),
                (2, 0, 0),
            ),
        )  # multiple rays
    ]
)
def test_compute_ray_intersection(euler_angles, position, ray_nodes, expected):
    """
    Tests for compute_ray_intersection
    """
    screen_center = (0, 0)  # not relevant
    pixel_size = 1  # not relevant

    screen = Screen(position, euler_angles, pixel_size, screen_center)

    # Compute and test intersection
    intersection = screen.compute_ray_intersection(ray_nodes)
    print(intersection, expected)

    if numpy.array(expected).size == 3:
        numpy.testing.assert_array_almost_equal(intersection[0], expected, decimal=3)
    else:
        numpy.testing.assert_array_almost_equal(intersection, expected, decimal=3)





