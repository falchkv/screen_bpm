import numpy


def compute_xy_angles(ray_nodes):
    """
    Computes the angle of the vertically and horizontally
    projected ray with respect to the z-axis, referred to as the x and y
    angels, respectively.

    Parameters
    ----------
    ray_nodes : numpy.ndarray, list or tuple
        Vectors that contain xyz coordinates of two points on the ray.

    Returns
    -------
    numpy.ndarray
        The x and y angles
    """
    # Convert to array in case of list or tuple.
    ray_nodes = numpy.array(ray_nodes)
    if ray_nodes.ndim == 1:
        ray_nodes = numpy.expand_dims(ray_nodes, axis=0)

    x1 = ray_nodes[:, 0]
    x2 = ray_nodes[:, 3]
    y1 = ray_nodes[:, 1]
    y2 = ray_nodes[:, 4]
    z1 = ray_nodes[:, 2]
    z2 = ray_nodes[:, 5]

    delta_x = x2 - x1
    delta_y = y2 - y1
    delta_z = z2 - z1
    angle_x = numpy.arctan(delta_x / delta_z)
    angle_y = numpy.arctan(delta_y / delta_z)

    return numpy.hstack([angle_x, angle_y])


def compute_beam_position(ray_nodes, z_beam_position):
    """
    Computes the position of intersection of rays with the plane
    z=z_beam_position.

    Parameters
    ----------
    ray_nodes : numpy.ndarray, list or tuple
        Vectors that contain xyz coordinates of two points on the ray.

    z_beam_position : float
        Z-coordinate of plane where beam intersection is to be computed.

    Returns
    -------
    numpy.ndarray
        The x and y coordinates of the intersections.
    """
    ray_nodes = numpy.array(ray_nodes)
    if ray_nodes.ndim == 1:
        ray_nodes = numpy.expand_dims(ray_nodes, axis=0)

    x1 = ray_nodes[:, 0]
    x2 = ray_nodes[:, 3]
    y1 = ray_nodes[:, 1]
    y2 = ray_nodes[:, 4]
    z1 = ray_nodes[:, 2]
    z2 = ray_nodes[:, 5]

    delta_z = z2 - z1

    y_intersect = (y2 - y1) * (z_beam_position - z1) / delta_z + y1
    x_intersect = (x2 - x1) * (z_beam_position - z1) / delta_z + x1

    return numpy.squeeze(numpy.vstack([x_intersect, y_intersect]).T)


def fit_ray(xyz_positions, sigmas=None):
    """
    Fits a straight line to a set of xyz-coordinates. The z-coordinate is
    treated as certain, the x and y positions are fitted.

    Parameters
    ----------
    xyz_positions : numpy.ndarray, list or tuple
        Array that contain measured xyz coordinates of the ray.


    sigmas : numpy.ndarray, optional
        An estimate of the uncertainty of the provided xyz_positions. These
        values are used to weigh the data points in the fitting.

    Returns
    -------
    numpy.ndarray
        Vector that contain xyz coordinates of two points on the ray. I.e.
        [x1, y1, z1, x2, y2, z2]
    """
    # Convert to ndarray if list or tuple.
    if isinstance(xyz_positions, (tuple, list)):
        xyz_positions = numpy.array(xyz_positions)

    # initialize weights.
    if sigmas is None:
        weights = None
    else:
        # Convert to ndarray if list or tuple.
        if isinstance(sigmas, (tuple, list)):
            sigmas = numpy.array(sigmas)

        weights = 1/sigmas * numpy.max(sigmas)

    x = xyz_positions[:, 0]
    y = xyz_positions[:, 1]
    z = xyz_positions[:, 2]
    poly_x = numpy.polynomial.polynomial.polyfit(z, x, 1, w=weights)
    poly_y = numpy.polynomial.polynomial.polyfit(z, y, 1, w=weights)

    z_min = numpy.min(z)
    z_max = numpy.max(z)
    ray_nodes = numpy.array([
        numpy.polynomial.polynomial.polyval(z_min, poly_x),
        numpy.polynomial.polynomial.polyval(z_min, poly_y),
        z_min,
        numpy.polynomial.polynomial.polyval(z_max, poly_x),
        numpy.polynomial.polynomial.polyval(z_max, poly_y),
        z_max,
    ])

    return ray_nodes
