import numpy
from scipy.spatial.transform import Rotation


class Screen:
    """
    Class for converting between pixel coordinates (u, v) and lab-coordinates
    (xyz).

    The canonical screen has its uv-coordinates at the origin, with u axis
    pointing in negative y direction, v-axis pointing in negative x direction,
    and its normal in the z-direction. The normal points away from the
    observer.

    The xyz coordinate of a pixel is found with following procedure:
    1: offset by screen center
        The screen center is now at the origin.

    2: scale by pixel_size
        The coordinates are now in meters.

    3: Rotation about origin by given euler angles
        Screen center is still in the origin, but the screen has been rotated.

    4: translate screen center to given position
        The screen center is now at the given position.
    """

    def __init__(self, position, euler_angles, pixel_size, screen_center):
        """
        Parameters
        ---------
        pixel_size : float or iterable containing 2 floats
            The pixel size of the detector in meters. (UxV)
        """
        # Convert pixel_size to tuple, if it is a float or an int.
        if isinstance(pixel_size, (float, int)):
            pixel_size = (pixel_size, pixel_size)

        elif isinstance(pixel_size, (list, tuple, numpy.ndarray)):
            # If only one element was given, assume square pixels.
            if len(pixel_size) == 1:
                pixel_size = (pixel_size[0], pixel_size[0])

            elif len(pixel_size) == 2:
                # If pixel_size is an ndarray of length 2, check that there
                # is only one dimension.
                if isinstance(pixel_size, numpy.ndarray):
                    if pixel_size.ndim != 1:
                        raise ValueError(
                            "pixel_size has invalid number of dimensions: {}"
                                .format(pixel_size.shape)
                        )

            elif len(pixel_size) > 2:
                raise ValueError(
                    "pixel_size contains too many elements: {}"
                        .format(pixel_size)
                )

        self.pixel_size = pixel_size
        self.rotation = Rotation.from_euler(
            'ZXY', euler_angles, degrees=False
        )
        self.position = position
        self.screen_center = screen_center

    def _uv_to_uv_prime(self, uv_points):
        """
        Converts uv-coordinates to uv_prime-coordinates. The uv_prime
        coordinate system is aligned with the uv-coordinate system, but may be
        scaled.

        Parameters
        ---------
        uv_points : numpy.ndarray, tuple or list
            Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u1,v1], ..., [un,vn]].

        Returns
        -------
        numpy.ndarray
            Uv_prime points. Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u'1,v'1], ..., [u'n,v'n]].
        """
        # Convert to ndarray if tuple or list.
        if isinstance(uv_points, (tuple, list)):
            uv_points = numpy.expand_dims(numpy.array(uv_points), axis=0)

        # Move screen center to uv-origin
        uv_points = uv_points - self.screen_center

        # Crate affine transformation matrix
        # This is currently a simple scaling, however it may be necessary to
        # allow sheering in the future.
        affine = numpy.array(
            [[self.pixel_size[0], 0],
             [0, self.pixel_size[1]]]
        )

        uv_prime_points = numpy.matmul(affine, uv_points.T).T

        return uv_prime_points

    def _uv_prime_to_xyz(self, uv_prime):
        """
        Converts uv_prime coordinates to xyz coordinates. See doc string of
        _uv_to_uv_prime() for a definition of uv_prime coordinates.

        Parameters
        ----------
        uv_prime : numpy.ndarray
            Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u'1,v'1], ..., [u'n,v'n]].

        Returns
        -------
        numpy.ndarray
            (n, 3) array of xyz coordinates. [[x1,y1,z1], ..., [xn,yn,zn]].
        """
        # Convert to ndarray if tuple or list.
        if isinstance(uv_prime, (tuple, list)):
            uv_prime = numpy.expand_dims(numpy.array(uv_prime), axis=0)

        # Expand dims if 1d array.
        if uv_prime.ndim == 1:
            uv_prime = numpy.expand_dims(uv_prime, axis=0)

        # Extend to third dimension by appending w=0.
        xyz_prime = numpy.hstack([
            -numpy.flip(uv_prime, axis=1),
            numpy.zeros((len(uv_prime), 1))
        ])

        # Apply rotation and add lab-coordinate position of the screen.
        xyz = self.rotation.apply(xyz_prime, inverse=False) + self.position

        return xyz

    def uv_to_xyz(self, uv_points):
        """
        Converts uv-coordinates to xyz coordinates.

        Parameters
        ---------
        uv_points : numpy.ndarray, tuple or list
            Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u1,v1], ..., [un,vn]].

        Returns
        -------
        numpy.ndarray
            (n, 3) array of xyz-coordinates. [[x1,y1,z1], ..., [xn,yn,zn]].
        """

        uv_prime = self._uv_to_uv_prime(uv_points)
        xyz = self._uv_prime_to_xyz(uv_prime)

        return xyz

    def _xyz_to_uv_prime(self, xyz):
        """
        Converts uv_prime coordinates to xyz coordinates. See doc string of
        _uv_to_uv_prime() for a definition of uv_prime coordinates.

        Parameters
        ----------
        xyz : numpy.ndarray, tuple or list
            Array of dimension (n, 3), where n is the number of points.
            I.e. it is of the form [[x1,y1,z1], ..., [xn,yn,zn]].

        Returns
        -------
        numpy.ndarray
            Uv_prime points. Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u'1,v'1], ..., [u'n,v'n]].
        """
        # Convert to ndarray if tuple or list.
        if isinstance(xyz, (tuple, list)):
            xyz = numpy.expand_dims(numpy.array(xyz), axis=0)

        # subtract lab-coordinate position of the screen.
        xyz = xyz - self.position

        # un-apply rotation
        xyz = self.rotation.apply(xyz, inverse=True)

        # Discard/project z-coordinate.
        # Invert axis to convert to uv_prime coordiantes
        uv_prime = numpy.flip(-xyz[:, :2], axis=1)

        # Expand dims if 1d array.
        if uv_prime.ndim == 1:
            uv_prime = numpy.expand_dims(uv_prime, axis=0)

        return uv_prime

    def _uv_prime_to_uv(self, uv_prime_points):
        """
        Converts uv_prime-coordinates to uv-coordinates. The uv_prime
        coordinate system is aligned with the uv-coordinate system, but may be
        scaled.

        Parameters
        ---------
        uv_prime_points : numpy.ndarray, tuple or list
            Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u1',v1'], ..., [un',vn']].

        Returns
        -------
        numpy.ndarray
            Uv points. Array of dimension (n, 2), where n is the number of points.
            I.e. it is of the form [[u1,v1], ..., [un,vn]].
        """
        # Convert to ndarray if tuple or list.
        if isinstance(uv_prime_points, (tuple, list)):
            uv_prime_points = numpy.expand_dims(numpy.array(uv_prime_points), axis=0)

        # Crate affine transformation matrix
        # This is currently a simple scaling, however it may be necessary to
        # allow sheering in the future.
        affine = numpy.array(
            [[self.pixel_size[0], 0],
             [0, self.pixel_size[1]]]
        )

        # apply inverse matrix
        uv_points = numpy.matmul(numpy.linalg.inv(affine), uv_prime_points.T).T

        # Undo move screen center to uv-origin
        uv_points = uv_points + self.screen_center

        return uv_points

    def xyz_to_uv(self, xyz_points):
        """
        Converts xyz-coordinates to uv coordinates.

        Parameters
        ---------
        xyz_points : numpy.ndarray, tuple or list
            Array of dimension (n, 3), where n is the number of points.
            I.e. it is of the form [[x1,y1,z1], ..., [xn,yn,zn]].

        Returns
        -------
        numpy.ndarray
            (n, 2) array of uv-coordinates. [[u1,v1], ..., [un,vn]].
        """

        uv_prime = self._xyz_to_uv_prime(xyz_points)
        uv = self._uv_prime_to_uv(uv_prime)

        return uv

    def compute_normal(self):
        """
        Computes the screen normal.

        Returns
        -------
        numpy.ndarray
            Array of dimension (3, ). A normalized vector indicating direction of screen normal.
        """
        unit_u = self.uv_to_xyz([1, 0])[0] - self.position
        unit_v = self.uv_to_xyz([0, 1])[0] - self.position

        normal = -numpy.cross(unit_u, unit_v)
        normal = normal/numpy.linalg.norm(normal)

        return normal

    def compute_ray_intersection(self, ray_nodes):
        """
        Compute intersection of ray with screen.

        Parameters
        ---------
        ray_nodes : numpy.ndarray, list or tuple
            Vectors that contain xyz coordinates of two points on the ray.

        Returns
        -------
        numpy.ndarray
            Array of dimension (n, 3), where n is the number of rays.
            I.e. it is of the form [[x1,y1,z1], ..., [xn,yn,zn]].
        """
        # Convert to array in case of list or tuple.
        ray_nodes = numpy.array(ray_nodes)
        if ray_nodes.ndim == 1:
            ray_nodes = numpy.expand_dims(ray_nodes, axis=0)

        point_1 = ray_nodes[:, :3]
        point_2 = ray_nodes[:, 3:6]

        difference = point_2 - point_1

        normal = self.compute_normal()

        # Equation to solve for s: (point_1 + s*difference)*normal = self.position*normal
        # s*difference*normal = -point_1*normal

        s = (numpy.dot(self.position, normal) - numpy.dot(point_1, normal)) / numpy.dot(difference, normal)

        intersection = point_1 + (difference.T*s).T

        return intersection

    @staticmethod
    def from_calibration(calibration):
        """
        Creates a Screen object from calibration dictionary.

        Parameters
        ----------
        calibration : dict
         Dictionary containing keys position, euler_angles, pixel_size,
         screen_center.

        Returns
        -------
        Screen
            A screen object.
        """
        screen = Screen(
            calibration['position'],
            calibration['euler_angles'],
            calibration['pixel_size'],
            calibration['screen_center'],
        )
        return screen
