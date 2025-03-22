import numpy as np

def find_subarray_indices(array, subarray):
    a = np.asarray(array)
    s = np.asarray(subarray)

    # Generate sliding window views
    shape = (a.size - s.size + 1, s.size)
    strides = (a.strides[0], a.strides[0])
    windows = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # Compare each window with subarray
    matches = np.all(windows == s, axis=1)

    # Return the matching indices
    return np.where(matches)[0]


if __name__ == "__main__":
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6]
    subarray = [4, 5, 6]
    indices = find_subarray_indices(array, subarray)
    print("Indices of subarray:", indices)