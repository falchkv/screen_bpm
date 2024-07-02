import numpy
from skimage.feature import blob_doh
from scipy.ndimage import median_filter


def extract_beam_position(image):
    """
    Extracts the beam position from an image of the beam using a median filter
    adn difference of hessian blob detection.

    Parameters
    ----------
    image : numpy.ndarray
        The image. Assumed to be monochromatic.

    Returns
    -------
    numpy.ndarray
        Pixel indices (vertical, horizontal) of the beam position.

    float
        Sigma of the beam blob.
    """

    if image.ndim > 2:
        raise ValueError('Too many image dimensions: {}'.format(image.ndim))

    # median filter
    image = median_filter(image, size=3)

    # Detect blobs
    blobs = blob_doh(
        image,
        min_sigma=3,
        max_sigma=1000,
        num_sigma=20,
        log_scale=True,
        threshold=0.01,
    )

    # return largest blob
    n_blobs, _ = blobs.shape

    if n_blobs == 1:
        blob_u = blobs[0, 0]
        blob_v = blobs[0, 1]
        blob_sigma = blobs[0, 2]
    elif n_blobs > 1:
        blob_sigmas = blobs[:, 2]
        largest_blob_index = numpy.argmax(blob_sigmas)
        blob_u = blobs[largest_blob_index, 0]
        blob_v = blobs[largest_blob_index, 1]
        blob_sigma = blobs[largest_blob_index, 2]
    else:
        raise ValueError('No blobs were found')

    return numpy.array([blob_u, blob_v]), blob_sigma


def extract_max_position(image, median_filter_size=3):
    """
    Extracts the pixel coordinate with the highest intensity. A median filter is applied first.

    Parameters
    ----------
    image : numpy.ndarray
        The image. Assumed to be monochromatic.

    median_filter_size : float, optional
        Size of median filter.

    Returns
    -------
    numpy.ndarray
        Pixel indices (vertical, horizontal) of the max intensity position.
    """

    if image.ndim > 2:
        raise ValueError('Too many image dimensions: {}'.format(image.ndim))

    # median filter
    image = median_filter(image, size=median_filter_size)

    u, v = numpy.where(image == numpy.max(image))
    u = numpy.mean(u)  # there may be multiple pixels with max value
    v = numpy.mean(v)  # there may be multiple pixels with max value

    return numpy.array([u, v])