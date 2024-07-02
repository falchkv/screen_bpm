import pytest
import numpy
from scipy.spatial.transform import Rotation

from screen_bpm.lm_screen_analysis.camera_operations import *


@pytest.mark.parametrize(
    "angle, axis, fx, fy, skew, principal_x, principal_y, camera_center",
    [
        (numpy.pi / 10, 0, 1.0, 1.0, 0.0, 0.0, 0.0, [1, 0, 0, 1.0]),
        (numpy.pi / 10, 1, 1.0, 1.0, 1.0, 0.0, 1.0, [0, 2, 0, 1.0]),
        (-numpy.pi / 10, 2, 1.0, 1.0, -40.0, 0.0, 2.0, [2, 0, -5, 1.0]),
        (numpy.pi / 10, 2, 3.0, 1.0, 0.0, 4.0, -6.0, [0, 3, 5, -1.0]),
    ]
)
def test_decompose_projection_matrix(angle, axis, fx, fy, skew, principal_x, principal_y, camera_center):
    """
    Tests for decompose_projection_matrix()
    """
    camera_matrix = numpy.array([
        [fx, skew, principal_x],
        [0, fy, principal_y],
        [0, 0, 1.0],
    ])
    euler_angles = [0, 0, 0]
    euler_angles[axis] = angle
    rotation_matrix = Rotation.from_euler(
            'XYZ', euler_angles, degrees=False
        ).as_matrix()
    camera_center = numpy.array([camera_center]).T
    projection_matrix = synthesize_projection_matrix(camera_matrix, rotation_matrix, camera_center)
    K, R, c = decompose_projection_matrix(projection_matrix)

    numpy.testing.assert_array_almost_equal(R, rotation_matrix, decimal = 10)
    numpy.testing.assert_array_almost_equal(K, camera_matrix, decimal=10)
    numpy.testing.assert_array_almost_equal(c/c[-1], camera_center/camera_center[-1], decimal=10)


@pytest.mark.parametrize(
    "angle, axis, fx, fy, skew, principal_x, principal_y, camera_center",
    [
        (numpy.pi / 10, 0, 1.0, 1.0, 0.0, 0.0, 0.0, [1, 0, 0, 1.0]),
        (numpy.pi / 10, 1, 1.0, 1.0, 1.0, 0.0, 1.0, [0, 2, 0, 1.0]),
        (-numpy.pi / 10, 2, 1.0, 1.0, -40.0, 0.0, 2.0, [2, 0, -5, 1.0]),
        (numpy.pi / 10, 2, 3.0, 1.0, 0.0, 4.0, -6.0, [0, 3, 5, -1.0]),
    ]
)
def test_rotation_correspondance(angle, axis, fx, fy, skew, principal_x, principal_y, camera_center):
    """
    Tests for decompose_projection_matrix()
    """
    camera_matrix = numpy.array([
        [fx, skew, principal_x],
        [0, fy, principal_y],
        [0, 0, 1.0],
    ])
    euler_angles = [0, 0, 0]
    euler_angles[axis] = angle
    rotation_matrix = Rotation.from_euler(
            'XYZ', euler_angles, degrees=False
        ).as_matrix()
    camera_center = numpy.array([camera_center]).T
    projection_matrix1 = synthesize_projection_matrix(camera_matrix, numpy.eye(3), camera_center)
    projection_matrix2 = rotate_camera(projection_matrix1, rotation_matrix)
    homography = camera_rotation_homography(projection_matrix1, rotation_matrix)
    n_points = 10
    X = numpy.random.rand(4, n_points)

    x1 = projection_matrix1 @ X
    x2 = projection_matrix2 @ X
    x2_from_homography = homography @ x1

    numpy.testing.assert_array_almost_equal(x2, x2_from_homography, decimal=10)


@pytest.mark.parametrize(
    "angle, axis, fx, fy, skew, principal_x, principal_y, camera_center",
    [
        (numpy.pi / 10, 0, 1.0, 1.0, 0.0, 0.0, 0.0, [1, 0, 0, 1.0]),
        (numpy.pi / 10, 1, 1.0, 1.0, 1.0, 0.0, 1.0, [0, 2, 0, 1.0]),
        (-numpy.pi / 10, 2, 1.0, 1.0, -40.0, 0.0, 2.0, [2, 0, -5, 1.0]),
        (numpy.pi / 10, 2, 3.0, 1.0, 0.0, 4.0, -6.0, [0, 3, 5, -1.0]),
    ]
)
def test_rotate_projection_plane(angle, axis, fx, fy, skew, principal_x, principal_y, camera_center):
    """
    Tests for decompose_projection_matrix()
    """
    camera_matrix = numpy.array([
        [fx, skew, principal_x],
        [0, fy, principal_y],
        [0, 0, 1.0],
    ])
    euler_angles = [0, 0, 0]
    euler_angles[axis] = angle
    rotation_matrix = Rotation.from_euler(
            'XYZ', euler_angles, degrees=False
        ).as_matrix()
    camera_center = numpy.array([camera_center]).T
    projection_matrix1 = synthesize_projection_matrix(camera_matrix, numpy.eye(3), camera_center)
    projection_matrix2 = rotate_projection_plane(projection_matrix1, rotation_matrix)
    homography = projection_plane_rotation_homography(projection_matrix1, rotation_matrix)
    n_points = 100
    X = numpy.random.rand(4, n_points)

    x1 = projection_matrix1 @ X
    x2 = projection_matrix2 @ X
    x2_from_homography = homography @ x1

    numpy.testing.assert_array_almost_equal(x2, x2_from_homography, decimal=10)


@pytest.mark.parametrize(
    "sx, sy",
    [
        (0, 0),
        (1, 1),
        (-100.0, 30.),
    ]
)
def test_get_xy_shearing_homography(sx, sy):
    """
    Tests for decompose_projection_matrix()
    """
    n_points = 100
    X0 = numpy.random.rand(4, n_points)
    X0_on_plane = numpy.random.rand(4, n_points)
    X0_on_plane[2, :] = 0

    x0 = X0[0, :] / X0[-1, :]
    y0 = X0[1, :] / X0[-1, :]
    z0 = X0[2, :] / X0[-1, :]
    x0_on_plane = X0_on_plane[0, :] / X0_on_plane[-1, :]
    y0_on_plane = X0_on_plane[1, :] / X0_on_plane[-1, :]
    z0_on_plane = X0_on_plane[2, :] / X0_on_plane[-1, :]

    shearing_homography = get_xy_shearing_homography(sx, sy)
    X1 = shearing_homography @ X0
    X1_on_plane = shearing_homography @ X0_on_plane

    x1 = X1[0, :] / X1[-1, :]
    y1 = X1[1, :] / X1[-1, :]
    z1 = X1[2, :] / X1[-1, :]
    x1_on_plane = X1_on_plane[0, :] / X1_on_plane[-1, :]
    y1_on_plane = X1_on_plane[1, :] / X1_on_plane[-1, :]
    z1_on_plane = X1_on_plane[2, :] / X1_on_plane[-1, :]

    x1_expected = x0 + z0 * sx
    y1_expected = y0 + z0 * sy
    z1_expected = z0

    # Assert that points on plane did not change
    numpy.testing.assert_array_almost_equal(X1_on_plane/X1_on_plane[-1, :], X0_on_plane/X0_on_plane[-1, :])

    # Assert that points moved as expected
    numpy.testing.assert_array_almost_equal(x1, x1_expected)
    numpy.testing.assert_array_almost_equal(y1, y1_expected)
    numpy.testing.assert_array_almost_equal(z1, z1_expected)


@pytest.mark.parametrize(
    "cam_center, target_cam_center, plane",
    [
        ([0, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 0]),
        ([0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0]),
        ([0, 1, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]),
    ]
)
def test_get_shearing_homography(cam_center, target_cam_center, plane):
    """
    Tests for decompose_projection_matrix()
    """
    plane = numpy.array(plane)
    n_points = 2
    X0 = numpy.random.rand(4, n_points)
    on_plane_basis = generate_points_on_planes(plane)
    # Create random combinations of points to create new random points, still on plane.
    X0_on_plane = on_plane_basis @ numpy.random.rand(3, n_points)

    # create cam_plane
    cam_plane = plane.copy()
    cam_plane[-1] -= numpy.sum(plane * cam_center)/cam_center[-1]
    # Check that cam is in fact on cam plane
    numpy.testing.assert_array_almost_equal(
        numpy.sum(cam_plane * cam_center), 0, decimal=10
    )
    # Create random points on cam plane.
    X0_on_cam_plane = on_plane_basis @ numpy.random.rand(3, n_points)

    x0_on_plane = X0_on_plane[0, :] / X0_on_plane[-1, :]
    y0_on_plane = X0_on_plane[1, :] / X0_on_plane[-1, :]
    z0_on_plane = X0_on_plane[2, :] / X0_on_plane[-1, :]
    x0_on_cam_plane = X0_on_cam_plane[0, :] / X0_on_cam_plane[-1, :]
    y0_on_cam_plane = X0_on_cam_plane[1, :] / X0_on_cam_plane[-1, :]
    z0_on_cam_plane = X0_on_cam_plane[2, :] / X0_on_cam_plane[-1, :]

    shearing_homography = get_shearing_homography(cam_center, target_cam_center, plane)
    X1_on_plane = shearing_homography @ X0_on_plane
    X1_on_cam_plane = shearing_homography@X0_on_cam_plane

    # Assert that points on plane did not change
    numpy.testing.assert_array_almost_equal(X1_on_plane/X1_on_plane[-1, :], X0_on_plane/X0_on_plane[-1, :])

    # Assert that points on cam plane have same distances
    x1_on_cam_plane = X1_on_cam_plane[0, :] / X1_on_cam_plane[-1, :]
    y1_on_cam_plane = X1_on_cam_plane[1, :] / X1_on_cam_plane[-1, :]
    z1_on_cam_plane = X1_on_cam_plane[2, :] / X1_on_cam_plane[-1, :]
    dx0 = numpy.sqrt(numpy.subtract.outer(x0_on_cam_plane, x0_on_cam_plane))
    dy0 = numpy.sqrt(numpy.subtract.outer(y0_on_cam_plane, y0_on_cam_plane))
    dz0 = numpy.sqrt(numpy.subtract.outer(z0_on_cam_plane, z0_on_cam_plane))
    dx1 = numpy.sqrt(numpy.subtract.outer(x1_on_cam_plane, x1_on_cam_plane))
    dy1 = numpy.sqrt(numpy.subtract.outer(y1_on_cam_plane, y1_on_cam_plane))
    dz1 = numpy.sqrt(numpy.subtract.outer(z1_on_cam_plane, z1_on_cam_plane))
    numpy.testing.assert_array_almost_equal(dx0, dx1, decimal=10)
    numpy.testing.assert_array_almost_equal(dy0, dy1, decimal=10)
    numpy.testing.assert_array_almost_equal(dz0, dz1, decimal=10)

    # Assert that cam center went to the right place
    result_cam_center = shearing_homography @ numpy.expand_dims(cam_center, axis=1)
    numpy.testing.assert_array_almost_equal(
        result_cam_center.flatten(), numpy.array(target_cam_center)
    )


@pytest.mark.parametrize(
    "initial_basis",
    [
        (numpy.array([[1, 0, 0]]).T),
        (numpy.array([[1, 0, 0], [0, 0, 1]]).T),
        (numpy.array([[1, 0, 0, 0], [0, 0, 0, 1]]).T),
    ]
)
def test_complete_basis(initial_basis):
    additional_basis = complete_basis(initial_basis)

    products = initial_basis.T @ additional_basis
    numpy.testing.assert_array_almost_equal(products, numpy.zeros(products.shape), decimal=10)


@pytest.mark.parametrize(
    "planes",
    [
        (numpy.array([[1, 0, 0, 0]]).T),
        (numpy.array([[1, 0, 0, 0], [0, 0, 1, 2]]).T),
        (numpy.array([[1, 0, 5, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).T),
    ]
)
def test_complete_basis(planes):
    points = complete_basis(planes)

    # assert that all points are on all of the planes.
    # if product between point and plane is 0, point is on the plane.
    products = planes.T @ points
    numpy.testing.assert_array_almost_equal(products, numpy.zeros(products.shape), decimal=10)
