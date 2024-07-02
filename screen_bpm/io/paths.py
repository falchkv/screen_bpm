import os
import glob
import re

BASE_PATHS = [
    '/gpfs/current',
    '/gpfs/commissioning',
    '/asap3/petra3/gpfs/p06/*/data',
    '/asap3/petra3/gpfs/p06/*/commissioning',
    'U:/p06/*/data',
    'U:/p06/*/commissioning',
    '/run/user/1001/gvfs/sftp:host=max-fs-display.desy.de,user=falchkv/asap3/petra3/gpfs/p06/*/data',
    '/run/user/1001/gvfs/sftp:host=max-fs-display.desy.de,user=falchkv/asap3/petra3/gpfs/p06/*/commissioning',
    '/run/user/1001/gvfs/sftp:host=max-fsc.desy.de,user=falchkv/asap3/petra3/gpfs/p06/*/data',
    '/run/user/1001/gvfs/sftp:host=max-fsc.desy.de,user=falchkv/asap3/petra3/gpfs/p06/*/commissioning',

]


def get_existing_base_path():
    """
    Returns base path that exists, in the sense that os.path.exists(path)
    returns true. Wild cards are kept as wild cards.

    Returns
    str
        Existing base path.
    """
    for base_path in BASE_PATHS:
        file_list = glob.glob(base_path)
        if len(file_list) > 0:
            existing_base_path = base_path
            break

    return existing_base_path


def base_name_replace(path):
    """
    Replaces the base name in path, with one accessible from current system.

    Parameters
    ----------
    path : str
        path whose base_name should be replaced
    """
    # Find base path in path.
    base_path_in_path = None
    for base_path in BASE_PATHS:
        # Use regular expressions to match path to existing_base_path, which
        # contains wildcards.
        pattern = base_path.replace('*', '.+')
        matches = re.match(pattern, path)

        # If there is a match. This is the base path in path.
        if matches:
            base_path_in_path = matches.group(0)
            break
    assert base_path_in_path is not None, 'No base path found in %s' % path

    # Find existing base path.
    existing_base_path = get_existing_base_path()

    # Replace base path with existing one.
    new_path = path.replace(base_path_in_path, existing_base_path)

    return new_path


def search_expid_path(experiment_id, base_path):
    """
    Searches for directories corresponding to the given experiment id in the
    base_path directory

    Parameters
    ----------
    experiment_id : str
        The experiment id.

    base_path : str
        Directory to search for experiment directory inside of.

    Returns
    -------
    str
        Path to experiment directory.
    """
    if 'current' in base_path or 'commissioning' in base_path:
        experiment_paths = glob.glob(os.path.join(base_path))
    else:
        experiment_paths = glob.glob(os.path.join(base_path, experiment_id))
    match = None
    for path in experiment_paths:
        if experiment_id in path:
            if os.path.isdir(path):
                match = path
                break

    return match


def experiment_path_from_id(experiment_id):
    """
    Searches for directories corresponding to the given experiment id.

    Parameters
    ----------
    experiment_id : str
        The experiment id.

    Returns
    -------
    str
        Path to experiment directory.
    """
    experiment_path = None

    for bp in BASE_PATHS:
        experiment_path = search_expid_path(experiment_id, bp)
        if experiment_path is not None:
            break

    return experiment_path


def search_for_key_file():
    """
    Tries to find key file in default locations.
    Returns None if no file was found.

    Returns
    -------
    str
        Path to key file.
    """
    default_key_file_paths = [
        os.path.join(os.path.expanduser('~'), '.config/DESY/LogbookHandler/keys.conf'),
        "keys.conf"
    ]
    key_file = None  # initiate
    for path in default_key_file_paths:
        file_list = glob.glob(path)
        if len(file_list) == 1:
            key_file = file_list[0]
        elif len(file_list) == 0:
            pass
        elif len(file_list) > 1:
            print('Multiple key files found. This should never happen.')
            print(file_list)

    if key_file is None:
        print(
            'Warning, no keys.conf file was found. '
            'Offline metadata will not be available.'
        )

    return key_file


def get_scan_paths(experiment_path, scan_ids=None, raw=True):
    """
    Parameters
    ----------
    experiment_path : str
        Path to experiment directory.

    scan_ids : list or in
        Scan ids to search for. If None, all scans are included.

    raw : bool
        If true, get the raw scan directories. If false, the processed
        directories is returned instead.

    Returns
    -------
    list
        List of scan directory paths.
    """
    # If scan_ids is a single int, make it a list of one element.
    if isinstance(scan_ids, int):
        scan_ids = [scan_ids]

    if raw:
        raw_or_processed = 'raw'
    else:
        raw_or_processed = 'processed'

    # Search for all scan directories files
    file_list = glob.glob(os.path.join(
        experiment_path, raw_or_processed, 'scan_*'
    ))
    file_list += glob.glob(os.path.join(
        experiment_path, raw_or_processed, '*', 'scan_*'
    ))

    # Keep only directories.
    file_list = [path for path in file_list if os.path.isdir(path)]

    # Sort
    file_list = sorted(file_list)

    # Filter the list if scan ids are specified.
    if scan_ids is not None:
        file_list = list(filter(
            lambda x:
            int(x.split('_')[-1].split('.')[0]) in scan_ids, file_list
        ))

    return file_list


def get_fluo_roi_image_path(experiment_path, scan_id, detector_name):
    """
    Gets the path to nxs file with roi images.

    Parameters
    ---------
    experiment_path : str
        Path to experiment directory.

    scan_id : int
        The scan_id.

    detector_name : int
        The name of the detector id. e.g. ardesia_01_all.

    Returns
    -------
    str
        Path to nxs file
    """
    scan_path = get_scan_paths(experiment_path, scan_ids=scan_id, raw=False)[0]

    nxs_path = os.path.join(
        scan_path,
        'roi_images_{}'.format(detector_name),
        'scan_%.5i_ROI_images.nxs' % (scan_id)
    )
    if not os.path.exists(nxs_path):
        raise IOError('Could not find {}'.format(nxs_path))

    return nxs_path
