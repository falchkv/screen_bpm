import numpy


class HomographyScreen:
    """
    Class for converting between pixel coordinates (u, v) and lab-coordinates
    (xyz).
    """

    def __init__(self, z_position, homogrpahy):
        """
        Parameters
        ---------
        z_position : float
            Screen position along optical axis

        homography : numpy.ndarray
            3x3 matrix converting camera image coordinates uv to xy lab-coordinates at z_position.
        """
        self.z_position = z_position
        self.homography = homogrpahy

    def convert_xyz_input(self, xyz):
        """
        Cponverts xyz to a numpy.ndarray with shape (n_points, 3).

        Parameters
        ----------
        xyz : list, tuple, numpy.ndarray
            xyz coordinates.

        Returns
        -------
        numpy.ndarray
            xyz coordinates with shape (n_points, 3).
        """
        # Convert to numpy.ndarray
        if isinstance(xyz, (list, tuple)):
            xyz = numpy.array(xyz)

        # Expand dims if there is only one point.
        if xyz.ndim == 1:
            xyz = numpy.expand_dims(xyz, axis=0)

        n_points, n_entries = xyz.shape
        if n_entries != 3:
            raise ValueError('wrong number of entries in xyz: {:n}. 3 is required.'.format(n_entries))

        return xyz

    def convert_uv_input(self, uv):
        """
        Cponverts uv to a numpy.ndarray with shape (n_points, 2).

        Parameters
        ----------
        uv : list, tuple, numpy.ndarray
            xyz coordinates.

        Returns
        -------
        numpy.ndarray
            uv coordinates with shape (n_points, 2).
        """
        # Convert to numpy.ndarray
        if isinstance(uv, (list, tuple)):
            uv = numpy.array(uv)

        # Expand dims if there is only one point.
        if uv.ndim == 1:
            uv = numpy.expand_dims(uv, axis=0)

        n_points, n_entries = uv.shape
        if n_entries != 2:
            raise ValueError('wrong number of entries in xyz: {:n}. 2 is required.'.format(n_entries))

        return uv

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
        uv_points = self.convert_uv_input(uv_points)

        n_points, _ = uv_points.shape
        uv_hom = numpy.vstack([uv_points.T, numpy.ones((n_points, ))])
        xy_hom = self.homography @ uv_hom
        xy = xy_hom[:2, :] / xy_hom[-1, :]
        xyz = numpy.vstack([xy, self.z_position * numpy.ones((n_points, ))]).T

        return xyz

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
        xyz_points = self.convert_xyz_input(xyz_points)
        n_points, _ = xyz_points.shape
        xy = xyz_points[:, :2]
        xy_hom = numpy.vstack([xy.T, numpy.ones((n_points, ))])

        uv_hom = numpy.linalg.inv(self.homography) @ xy_hom
        uv = uv_hom[:2, :] / uv_hom[-1, :]
        uv = uv.T
        return uv

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
        dpoint_dz = ((point_2 - point_1).T / (point_2[:, 2] - point_1[:, 2])).T
        Delta_z = (self.z_position - point_1[:, 2])
        intersection = point_1 + (dpoint_dz.T * Delta_z).T
        return intersection
