import h5py


def load_lm_screen_image(file_path, screen_name):
    with h5py.File(file_path) as h5:
        data = h5['entry/instrument/%s/data' % screen_name][()]
    return data


def load_lm_screen_images(file_path, screen_names):
    images = {}
    with h5py.File(file_path) as h5:
        for screen_name in screen_names:
            image = h5['entry/instrument/%s/data' % screen_name][()]
            images[screen_name] = image

    return images
