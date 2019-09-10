import numpy as np

# dictionary specifying axes of arrays to use for correct jittering of 3d arrays
jitter_axes_3d = {'j1': None,
                  'j2': (0, 1),
                  'j3': (0, 1),
                  'j4': (0, 1),
                  'j5': 0,
                  'j6': 1}

# dictionary specifying axes of arrays to use for correct jittering of 3d arrays
jitter_axes_4d = {'j1': None,
                  'j2': (1, 2),
                  'j3': (1, 2),
                  'j4': (1, 2),
                  'j5': 1,
                  'j6': 2}

# Dictionary with allowed jittering operations (rotations and flipping)
tasks = {}
task = lambda f: tasks.setdefault(f.__name__, f)


@task
def j1(img, axes):
    # Identity transform
    # print('Identity transform')
    return img.astype(np.float32)


@task
def j2(img, axes):
    # Rotate 90 degrees clockwise
    # print('Rotate 90 degrees clockwise')
    return np.rot90(img, k=3, axes=axes).astype(np.float32)


@task
def j3(img, axes):
    # Rotate 90 degrees counter-clockwise
    # print('Rotate 90 degrees counter-clockwise')
    return np.rot90(img, k=1, axes=axes).astype(np.float32)


@task
def j4(img, axes):
    # Rotate 180 degrees
    # print('Rotate 180 degrees')
    return np.rot90(img, k=2, axes=axes).astype(np.float32)


@task
def j5(img, axes):
    # Flip along height dimension
    # print('Flip along height dimension')
    return np.flip(img, axis=axes).astype(np.float32)


@task
def j6(img, axes):
    # Flip along width dimension
    # print('Flip along width dimension')
    return np.flip(img, axis=axes).astype(np.float32)
