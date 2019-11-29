import os
from pydoc import locate


def parse_classname(classname):
    return locate(classname)


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_common_shape(shape1, shape2):
    """ Get a common shape that fits both shapes. Dimensions that differ in size are set to None.
        Example: [None, 20, 100, 50], [None, 20, 200, 50] -> [None, 20, None, 50]
    """
    if len(shape1) != len(shape2):
        raise ValueError("Can't compute common shape. Ndims is different.")

    common_shape = [dim1 if dim1==dim2 else None for dim1, dim2 in zip(shape1, shape2)]

    return common_shape
