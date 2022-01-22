from typing import List


def array_to_integer(array: List[int]) -> int:
    """
    normally our evaluation is represented by a 5-integer array
    each item represents whether there is a partial (1) or full (2) match of the guess's letter i
    0 denotes absence
    we will convert this to an integer
    This integer is guaranteed to be between 0 and 4**5
    """
    assert isinstance(array, list)
    assert len(array) == 5
    v = 0
    for i, pos_value in enumerate(array):
        v += (4 ** i) * pos_value
    return v


def integer_to_arr(rval: int):
    arr = [0] * 5
    for i in range(5, -1, -1):
        # the number at position i
        # should be a value between 0-3
        if rval >= (4 ** i):
            rem = rval % (4 ** i)
            pos_value = int((rval - rem) / (4 ** i))
            arr[i] = pos_value
            rval -= arr[i] * (4 ** i)
    return arr