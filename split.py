import numpy as np
import random

def shuffle_arrays(arrays, set_seed=None):
    # copied from https://stackoverflow.com/a/51526109
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    #for arr in arrays:
    #    shuffle_array(arr, set_seed=set_seed)
    seed = np.random.randint(
        0, 2**(32 - 1) - 1) if set_seed is None else set_seed
    for arr in arrays:
        shuffle_array(arr, set_seed=seed)


def shuffle_array(arr, set_seed=None):
    #import numpy as np
    #seed = np.random.randint(
    #    0, 2**(32 - 1) - 1) if set_seed is None else set_seed
    rstate = np.random.RandomState(set_seed)
    rstate.shuffle(arr)

def shuffle_imu(arrays, set_seed=None, ignoreTop=False): # [(frames, 13, 120, 6, 1), (frames, 13)]
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    top = 1 if ignoreTop else 0
    for id_frame in range(arrays[0].shape[0]):
        seed = random.randint(1,2021)
        #[np.random.RandomState(seed).shuffle(arrays[id_arr][id_frame]) for id_arr in range(len(arrays))]
        [np.random.RandomState(seed).shuffle(arrays[id_arr][id_frame][top:]) for id_arr in range(len(arrays))]

if __name__ == "__main__":
    nb_imus = 5
    block_size = 2
    dims = 3
    nb_frames = 4
    x = np.arange(120).reshape((nb_frames, nb_imus, block_size, dims))
    y = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2]])
    rot_y = np.copy(y)
    print(x)
    print(y)
    
    print("shuffle")
    shuffle_arrays([x, y, rot_y], set_seed=1) # shuffle frames
    shuffle_imu([x, y], set_seed=1) # shuffle IMUs
    print(x[0])
    print(y[0])