import numpy
import cv2 as cv


def translate_camera(projection_matrix, translation_vector):
    """
    Returns the projection matrix corresponding to a camera that is translated relative to the input camera.

    Parameters
    ----------
    projection_matrix : numpy.ndarray
        (3, 4) projection matrix of input camera.

    translation_vector : numpy.ndarray
        (3, ) vector matrix, describing translation of output camera, relative to input camera.

    Returns
    -------
    numpy.ndarray
        Translated (3, 4) projection matrix.
    """

    camera_matrix, rotation_matrix, input_camera_center = decompose_projection_matrix(projection_matrix)
    if translation_vector.ndim == 1:
        translation_vector = numpy.expand_dims(translation_vector, axis=1)
    output_camera_center = input_camera_center
    output_camera_center[:3] += translation_vector * output_camera_center[-1]
    output_projection_matrix = synthesize_projection_matrix(camera_matrix, rotation_matrix, output_camera_center)
    return output_projection_matrix


def rotate_camera(projection_matrix, rotation_matrix):
    """
    Returns the projection matrix corresponding to a camera that is rotated relative to the input camera.

    Parameters
    ----------
    projection_matrix : numpy.ndarray
        (3, 4) projection matrix of input camera.

    rotation_matrix : numpy.ndarray
        (3, 3) rotation matrix, describing rotation of output camera, relative to input camera.

    Returns
    -------
    numpy.ndarray
        Rotated (3, 4) projection matrix.
    """
    camera_matrix, input_rotation_matrix, camera_center = decompose_projection_matrix(projection_matrix)
    output_rotation_matrix = rotation_matrix @ input_rotation_matrix
    output_projection_matrix = synthesize_projection_matrix(camera_matrix, output_rotation_matrix, camera_center)

    return output_projection_matrix


def camera_rotation_homography(projection_matrix, rotation_matrix):
    """
    Returns an homography that maps projected points on the input camera to projected points to a camera that is rotated
    relative to the input camera.

    Parameters
    ----------
    projection_matrix : numpy.ndarray
        (3, 4) projection matrix of input camera.

    rotation_matrix : numpy.ndarray
        (3, 3) rotation matrix, describing rotation of output camera, relative to input camera.

    Returns
    -------
    numpy.ndarray
        Rotated (3, 3) homography so that x_output = homography @ x_input, where x_output and x_input are the image
        coordinates of the input and output cameras, respectively, in homogenous coordinates.
    """
    camera_matrix, input_rotation_matrix, camera_center = decompose_projection_matrix(projection_matrix)
    homography = camera_matrix @ rotation_matrix @ numpy.linalg.inv(camera_matrix)

    return homography


def decompose_projection_matrix(projection_matrix):
    """
    Decomposes a projection matrix into its intrinsic and extrinsic parts.

    Parameters
    ----------
    projection_matrix : numpy.ndarray
        (3, 4) projection matrix

    Returns
    -------
    numpy.ndarray
        (3, 3) camera matrix, often denoted K in literature.

    numpy.ndarray
        (3, 3) rotation matrix, often denoted R in literature.

    numpy.ndarray
        (4, 1) vector. The 3D camera center, in homogeneous coordinates.
    """
    camera_matrix, rotation_matrix, camera_center, rotMatrX, rotMatrY, rotMatrZ, eulerAngles = cv.decomposeProjectionMatrix(projection_matrix)

    return camera_matrix, rotation_matrix, camera_center


def synthesize_projection_matrix(camera_matrix, rotation_matrix, camera_center):
    """
    Produces a projection matrix out of extrinsic and extrinsic parts.

    Parameters
    ----------
    camera_matrix : numpy.ndarray
        (3, 3) camera matrix, often denoted K in literature.

    rotation_matrix : numpy.ndarray
        (3, 3) rotation matrix, often denoted R in literature.

    camera_center : numpy.ndarray
        (4, 1) vector. The 3D camera center, in homogeneous coordinates.

    Returns
    -------
    numpy.ndarray
        (3, 4) projection matrix
    """
    camera_center = numpy.array(camera_center)
    if camera_center.ndim == 1:
        camera_center = numpy.expand_dims(camera_center, axis=1)
    initial = numpy.hstack([numpy.eye(3), -camera_center[:3]/camera_center[3]])
    projection_matrix = camera_matrix @ rotation_matrix @ initial
    return projection_matrix


def rotate_projection_plane(projection_matrix, rotation_matrix):
    """
    Returns the projection matrix corresponding to a camera whose projection plane is rotated relative to that of the
    input camera. Points projected to the origin will still be projected to the origin.

    Parameters
    ----------
    projection_matrix : numpy.ndarray
        (3, 4) projection matrix of input camera.

    rotation_matrix : numpy.ndarray
        (3, 3) rotation matrix, describing rotation of projection plane.

    Returns
    -------
    numpy.ndarray
        (3, 4) output projection matrix.
    """
    rotated_projection_matrix = rotate_camera(projection_matrix, rotation_matrix)

    # Compute back projection of origin to plane at infinity
    M = projection_matrix[:, :3]
    origin_axis_at_infinity = numpy.hstack([numpy.linalg.inv(M) @ numpy.array([0, 0, 1]), 0.0])

    # Compute the projection of the principal_axis_at_infinity onto the rotated camera.
    tmp_origin_point = rotated_projection_matrix @ origin_axis_at_infinity
    tmp_origin_point = tmp_origin_point / tmp_origin_point[-1]  # normalize

    # Adjust the camera matrix so that the projection of the principal_axis_at_infinity becomes [0, 0, 1]
    x0 = tmp_origin_point[0]
    y0 = tmp_origin_point[1]
    kappa = numpy.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    output_projection_matrix = kappa @ rotated_projection_matrix

    return output_projection_matrix


def projection_plane_rotation_homography(projection_matrix, rotation_matrix):
    """
    Returns an homography that maps projected points on the input camera to projected points to a camera whose
    projection plane is rotated relative to that of the input camera. Points projected to the origin will still be
    projected to the origin.

    Parameters
    ----------
    projection_matrix : numpy.ndarray
        (3, 4) projection matrix of input camera.

    rotation_matrix : numpy.ndarray
        (3, 3) rotation matrix, describing rotation of output camera, relative to input camera.

    Returns
    -------
    numpy.ndarray
        (3, 3) homography so that x_output = homography @ x_input, where x_output and x_input are the image
        coordinates of the input and output cameras, respectively, in homogenous coordinates.
    """
    camera_matrix, input_rotation_matrix, camera_center = decompose_projection_matrix(projection_matrix)
    output_projection_matrix = rotate_projection_plane(projection_matrix, rotation_matrix)
    output_camera_matrix, output_rotation_matrix, output_camera_center = decompose_projection_matrix(
        output_projection_matrix
    )
    homography = output_camera_matrix @ rotation_matrix @ numpy.linalg.inv(camera_matrix)

    return homography


def get_xy_shearing_homography(sx, sy):
    """
    Returns a 4x4 homography that shares 3D points, in homogeneous coordinates, leaving points on the xy plane (z=0)
    unchanged.

    Parameters
    ----------
    sx : float
        Sharing parameter in x direction. (x_out = x_in + sx * z_in)

    sy :  float
        Sharing parameter in y direction. (y_out = y_in + sy * z_in)

    Returns
    -------
    numpy.ndarray
        4x4 homography that shares 3D points, in homogeneous coordinates.
    """
    H = numpy.eye(4)
    H[0, 2] = sx
    H[1, 2] = sy

    return H


def get_shearing_homography(cam_center, target_cam_center, invariant_plane):
    """
    Returns a 4x4 homography that shares 3D points, in homogeneous coordinates, leaving points on the xy plane (z=0)
    unchanged.

    Parameters
    ----------
    cam_center
    target_cam_center
    invariant_plane

    Returns
    -------
    numpy.ndarray
        4x4 homography that shares 3D points, in homogeneous coordinates.
    """
    cam_center = numpy.array(cam_center)
    # Normalize so that last entry of cam_center is 1
    #cam_center = cam_center / cam_center[-1]

    # normalize plane so that the first three entries is a unit normal
    #invariant_plane = invariant_plane / numpy.linalg.norm(invariant_plane[:3])
    #normal = invariant_plane[:3]

    # Get 3 points on the invariant plane
    points_on_planes = generate_points_on_planes(invariant_plane)

    # Compute camera to plane distance. Assumes normalized plane and cam_center.
    #cam_distance = numpy.sum(invariant_plane*cam_center)

    initial = numpy.hstack([
        points_on_planes,
        numpy.expand_dims(cam_center, axis=1)
    ])

    target = numpy.hstack([
        points_on_planes,
        numpy.expand_dims(target_cam_center, axis=1)
    ])

    # H @ initial = target
    # initial.T @ H.T = target.T

    print('spacer')
    print(numpy.round(initial, decimals=4))

    H = numpy.linalg.solve(initial.T, target.T).T


    print(numpy.round(H @ initial, decimals = 4))
    print(numpy.round(target, decimals = 4))

    return H


def generate_points_on_planes(planes):
    """
    Uses complete_basis to generate points on the given set of planes.

    Parameters
    ----------
    planes : numpy.array
        matrix of shape (n_spatial_dim + 1, n_planes). Planes are given in homgeneous notation.

    Returns
    -------
    numpy.ndarray
        The generated points (n_spatial_dim + 1, n_spatial_dim + 1 - n_planes).
    """
    points = complete_basis(planes)
    return points


def complete_basis(initial_basis):
    """
    Uses numpy.linalg.qr to produce a complete basis for the vector space, keeping vectors in the known basis.
    
    Parameters
    ----------
    initial_basis

    Returns
    -------
    numpy.ndarray
        The generated basis vectors. (n_dim, n_dim - n_provided).
    """
    initial_basis = numpy.array(initial_basis)
    if initial_basis.ndim == 1:
        initial_basis = numpy.expand_dims(initial_basis, axis=1)

    n_dim, n_provided = initial_basis.shape

    vectors = numpy.hstack([initial_basis, numpy.eye(n_dim)])
    Q, R = numpy.linalg.qr(vectors, mode='reduced')
    return Q[:, n_provided:]
