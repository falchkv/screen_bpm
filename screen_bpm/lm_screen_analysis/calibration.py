import copy

import numpy
from scipy.optimize import minimize

from screen_bpm.lm_screen_analysis.screen import *


def evaluate_screen_positions(ray_nodes, screens):
    """
    Evaluates the beam position for a set of ray nodes and screens.

    Parameters
    ----------
    ray_nodes : numpy.ndarray, list or tuple
        Vectors that contain xyz coordinates of two points on the ray.

    screens : dict
        Dictionary of Screen objects, with strings as keys.

    Returns
    -------
    numpy.ndarray
        Uv intersection points. Array of dimension (n, 2), where n is the
        number of points. I.e. it is of the form [[u1,v1], ..., [un,vn]].

    """
    n_rays, _ = ray_nodes.shape
    n_screens = len(screens)
    uv_intersections = numpy.zeros((n_screens, n_rays, 2))
    for i, key_item_pair in enumerate(screens.items()):
        screen_name, screen = key_item_pair

        # compute xyz intersection
        xyz_intersection = screen.compute_ray_intersection(ray_nodes)
        uv_intersection = screen.xyz_to_uv(xyz_intersection)
        uv_intersections[i, :, :] = uv_intersection

    return uv_intersections


def calibrations_to_screens(calibrations):
    """
    Creates a dictionary of screen objects based on dictionary of calibrations.

    Parameters
    ----------
    calibrations : dict
        Dictionary of calibrations.

    Returns
    -------
    dict
        Dictionary of screens.
    """
    screens = {}
    for screen_name, calibration in calibrations.items():
        screens[screen_name] = Screen.from_calibration(calibration)

    return screens


def compute_error(uv_measured, ray_nodes, screens):
    """
    Parameters
    ----------
    uv_measured : numpy.ndarray
        Known intersection points. Array of dimension (n, 2), where n is the
        number of points. I.e. it is of the form [[u1,v1], ..., [un,vn]].

    ray_nodes : numpy.ndarray, list or tuple
        Vectors that contain xyz coordinates of two points on the ray.

    screens : dict
        Dictionary of Screen objects, with strings as keys.

    Returns
    -------
    float
        The error.

    """
    uv_simulated = evaluate_screen_positions(ray_nodes, screens)

    if uv_simulated.shape != uv_measured.shape:
        raise ValueError(
            'Shape of uv_measured differs from shape of uv_simulated.'
        )

    error = numpy.sum((uv_simulated - uv_measured) ** 2)
    return error


def calibration_to_array(calibrations, calibration_mask):
    """
    Takes an array of calibration parameters and feeds the values into
    calibration dictionaries, using calibration_mask.

    Parameters
    ----------
    calibrations : dict
        A dictionary of calibrations.

    calibration_mask: dict
        A calibration dictionary, with every element in every vector replaced
        by a boolean. True signifies that the element is to be inserted from
        the calibration_array.

    Returns
    -------
     numpy.ndarray
        A calibration_array. An Array containing the unmasked parameters of the
        calibration.
    """
    calibration_array = numpy.zeros((0, ))

    for screen_name, calibration in calibrations.items():
        mask_dict = calibration_mask[screen_name]
        for key, values in calibration.items():
            # Convert values to tuple if not tuple, list, or numpy.ndarray
            if not isinstance(values, (tuple, list, numpy.ndarray)):
                values = (values,)

            for i, value in enumerate(values):
                if mask_dict[key][i]:
                    calibration_array = numpy.append(calibration_array, value)

    return calibration_array


def array_to_calibration(
        calibration_array,
        calibration_mask,
        calibrations_initial
):
    """
    Takes an array of calibration parameters and feeds the values into
    calibration dictionaries, using calibration_mask.

    Parameters
    ----------
    calibration_array : numpy.ndarray, tuple, or list
        Array containing the unmasked parameters of the calibration.

    calibration_mask: dict
        A calibration dictionary, with every element in every vector replaced
        by a boolean. True signifies that the element is to be inserted from
        the calibration_array.

    calibrations_initial : dict
        Dictionary of calibrations. These values are used where
        calibration_mask contains False.

    Returns
    -------
    dict
        Dictionary of calibrations.
    """
    calibrations = copy.deepcopy(calibrations_initial)
    for screen_name, calibration in calibrations.items():
        mask_dict = calibration_mask[screen_name]
        for key, values in calibration.items():
            # Convert values to tuple if not tuple, list, or numpy.ndarray
            if not isinstance(values, (tuple, list, numpy.ndarray)):
                values = (values,)

            new_values = numpy.array(values)
            for i, value in enumerate(new_values):
                if mask_dict[key][i]:
                    new_values[i] = calibration_array[0]
                    calibration_array = calibration_array[1:]

            calibration[key] = new_values

    return calibrations


def default_calibration_scaler():
    """
    Returns dictionary of scaling parameters tobe used in fit().

    Returns
    -------
    dict
        Dictionary of scaling default parameters. Dictionary has same structure
        as a calibration.
    """
    scaler = {
            'screen_center': (1e-1, 1e-1),
            'position': (1e0, 1e0, 1e0),
            'euler_angles': (1e-1, 1e-1, 1e-1),
            'pixel_size': (1e0, 1e0),
        }
    return scaler


def apply_calibration_scaler(calibration_array, scaler, calibration_mask):
    """
    Returns dictionary of scaling parameters tobe used in fit().

    Parameters
    ----------
    calibration_array : numpy.ndarray
        Array containing the unmasked parameters of the calibration.

    scaler : dict
        Dictionary of scaling default parameters. Dictionary has same structure
        as a calibration.

    calibration_mask: dict
        A calibration dictionary, with every element in every vector replaced
        by a boolean. True signifies that the element is to be inserted from
        the calibration_array.

    Returns
    -------
    calibration_array : numpy.ndarray
        Array containing the unmasked parameters of the calibration.
    """
    # Apply scaling to prevent numerical precision loss.
    count = 0
    calibration_array = calibration_array.copy()
    for screen_name, mask_dict in calibration_mask.items():
        for key, values in mask_dict.items():
            for i, value in enumerate(values):
                if mask_dict[key][i]:
                    calibration_array[count] = \
                        calibration_array[count]*scaler[key][i]
                    count += 1

    return calibration_array


def unapply_calibration_scaler(calibration_array, scaler, calibration_mask):
    """
    Returns dictionary of scaling parameters tobe used in fit().

    Parameters
    ----------
    calibration_array : numpy.ndarray
        Array containing the unmasked parameters of the calibration.

    scaler : dict
        Dictionary of scaling default parameters. Dictionary has same structure
        as a calibration.

    calibration_mask: dict
        A calibration dictionary, with every element in every vector replaced
        by a boolean. True signifies that the element is to be inserted from
        the calibration_array.

    Returns
    -------
    numpy.ndarray
        The un-scaled calibration_array.
    """
    # Apply scaling to prevent numerical precision loss.
    count = 0
    calibration_array = calibration_array.copy()
    for screen_name, mask_dict in calibration_mask.items():
        for key, values in mask_dict.items():
            for i, value in enumerate(values):
                if mask_dict[key][i]:
                    calibration_array[count] = \
                        calibration_array[count] / scaler[key][i]
                    count += 1

    return calibration_array


def fit(
    uv_measured,
    ray_nodes_initial,
    calibrations_initial,
    calibration_mask
):
    """
    Parameters
    ----------

    Returns
    -------
    """
    # Define scaling
    ray_node_scaling = 1e-5

    # Place unknowns ray_node values into array, and unscale.
    ray_unknown_columns = [0, 1, 3, 4]  # z coordinate assumed to be known.
    ray_unknowns = ray_nodes_initial[:, ray_unknown_columns]
    ray_unknowns = ray_unknowns/ray_node_scaling
    ray_unknowns_shape = ray_unknowns.shape
    ray_unknowns_size = ray_unknowns.size

    # Place unknowns calibration values into array, and unscale.
    calibration_unknowns = calibration_to_array(
        calibrations_initial, calibration_mask
    )
    calibration_scaler = default_calibration_scaler()
    calibration_unknowns = unapply_calibration_scaler(
        calibration_unknowns, calibration_scaler, calibration_mask
    )

    def x_to_structured(x):
        """
        Takes the input array used in scipy.optimize.minimize, and sorts the
        data into appropriate data structures. Scaling is handeled here.
        """
        # Isolate and unscale calibration parameters.
        calibration_array = x[ray_unknowns_size:]
        calibration_array = apply_calibration_scaler(
            calibration_array, calibration_scaler, calibration_mask
        )

        # Isolate and scale ray_node parameters.
        ray_node_inputs = numpy.array(x[:ray_unknowns_size]) * ray_node_scaling

        # Assemble ray nodes
        trial_ray_nodes = ray_nodes_initial.copy()
        trial_ray_nodes[:, ray_unknown_columns] = ray_node_inputs.reshape(
            ray_unknowns_shape
        )

        # Apply scaling to prevent numerical precision loss.
        count = 0

        for screen_name, mask_dict in calibration_mask.items():
            for key, values in mask_dict.items():
                for i, value in enumerate(values):
                    if mask_dict[key][i]:
                        calibration_array[count] = calibration_array[count] * \
                                                   calibration_scaler[key][i]

        # Assemble screen calibration dictionaries
        trial_calibrations = array_to_calibration(
            calibration_array,
            calibration_mask,
            calibrations_initial,
        )

        return trial_ray_nodes, trial_calibrations

    def error_wrapper(x, uv_known):
        """
        Wrapper for error function, callable from scipy.optimize.minimize.
        """
        # Unpack x
        trial_ray_nodes, trial_calibrations = x_to_structured(x)

        # Create screen objects
        screens = calibrations_to_screens(trial_calibrations)

        # Compute and return error
        return compute_error(uv_known, trial_ray_nodes, screens)

    # Compute and display initial error
    initial_error = compute_error(
        uv_measured,
        ray_nodes_initial,
        calibrations_to_screens(calibrations_initial)
    )
    print('initial error: %.3e' % initial_error)

    def print_error(xk):
        """
        Callback function for scipy.optimize.minimize()
        """
        err = error_wrapper(xk, uv_measured)
        print(
            'Remaining error / initial error: %.3e' %
            (err/initial_error)
        )

    args = (uv_measured,)
    x0 = numpy.hstack([
        ray_unknowns.flatten(),
        calibration_unknowns.flatten(),
    ])
    minimization_result = minimize(
        error_wrapper,
        x0=x0,
        args=args,
        #  method='CG',  # Don't use CG.
        options={
            'disp': True  # Print convergence messages
        },
        callback=print_error,
    )

    ray_nodes, calibrations = x_to_structured(minimization_result.x)

    return ray_nodes, calibrations, minimization_result
