import numpy
import cv2 as cv


def find_homography(xy_in, xy_out):
    """
    This is a wrapper for cv.findHomography. It finds the homography, h, that satisfies xy_out_hom = h @ xy_in_hom,
    where xy_out_hom and xy_out_hom are homogeneous coordinates corresponding to xy_in, xy_out, which are given in
    Euclidean coordinates.

    Parameters
    ----------
    xy_in : numpy.ndarray
        Input xy positions in Euclidean coordinates. The shape is (2, n_points).
        
    xy_out
        Output xy positions in Euclidean coordinates. The shape is (2, n_points).
    Returns
    -------
    numpy.ndarray
        The homography.
    """
    h, status = cv.findHomography(
        xy_in.astype(numpy.float32).T,
        xy_out.astype(numpy.float32).T
    )
    return h