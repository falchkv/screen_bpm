import os

import pytest

from screen_bpm.lm_screen_analysis.image_operations import *
from p06io.images.image import Image

TEST_DATA_PATH = os.path.join('tests', 'test_data', 'lm_screen_analysis')


@pytest.mark.parametrize(
    "image_path, expected",
    [
        (os.path.join(TEST_DATA_PATH, '5000_ev', 'lm2_20000.png'), (168, 299)),
        (os.path.join(TEST_DATA_PATH, '5000_ev', 'lm3_25000.png'), (222, 218)),
        (os.path.join(TEST_DATA_PATH, '5000_ev', 'lm4_40000.png'), (324, 291)),
        (os.path.join(TEST_DATA_PATH, '10000_ev', 'lm2_50000.png'), (163, 298)),
    ]
)
def test_extract_beam_position(image_path, expected):
    """
    Tests for extract_beam_position
    """
    image = Image().read(image_path)

    # if color image, use only green channel
    if image.ndim == 3:
        image = image[:, :, 1]

    beam_position, sigma = extract_beam_position(image)

    error = numpy.abs(beam_position - numpy.array(expected))
    assert numpy.max(error) < 10
    assert isinstance(sigma, float)


@pytest.mark.parametrize(
    "image_path, expected",
    [
        (os.path.join(TEST_DATA_PATH, '5000_ev', 'lm2_20000.png'), (168, 299)),
        (os.path.join(TEST_DATA_PATH, '5000_ev', 'lm3_25000.png'), (222, 218)),
        (os.path.join(TEST_DATA_PATH, '5000_ev', 'lm4_40000.png'), (324, 291)),
        (os.path.join(TEST_DATA_PATH, '10000_ev', 'lm2_50000.png'), (163, 298)),
    ]
)
def test_extract_max_position(image_path, expected):
    """
    Tests for extract_max_position
    """
    image = Image().read(image_path)

    # if color image, use only green channel
    if image.ndim == 3:
        image = image[:, :, 1]

    max_position = extract_beam_position(image)

    error = numpy.abs(max_position - numpy.array(expected))
    assert numpy.max(error) < 10