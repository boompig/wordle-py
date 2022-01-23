from typing import List
import os.path
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TABLE_PATH = os.path.join(BASE_DIR, 'data-parsed/possibilities-table-base-3.npy')


from play import UNSAFE_eval_guess
from parse_data import read_parsed_words


def array_to_integer(array: List[int]) -> int:
    """
    normally our evaluation is represented by a 5-integer array
    each item represents whether there is a partial (1) or full (2) match of the guess's letter i
    0 denotes absence
    we will convert this to an integer
    This integer is guaranteed to be between 0 and 3**5
    """
    assert isinstance(array, list)
    assert len(array) == 5
    v = 0
    for i, pos_value in enumerate(array):
        assert pos_value < 3 and pos_value >= 0
        v += (3 ** i) * pos_value
    assert v < 255
    return v


def integer_to_arr(rval: int):
    arr = [0] * 5
    for i in range(5, -1, -1):
        # the number at position i
        # should be a value between 0-3
        if rval >= (3 ** i):
            rem = rval % (3 ** i)
            pos_value = int((rval - rem) / (3 ** i))
            arr[i] = pos_value
            rval -= arr[i] * (3 ** i)
    return arr


def load_possibilities_table(words: List[str]) -> pd.DataFrame:
    """
    Return a dataframe
    The index will represent guesses
    The columns will represent answers
    """
    table = np.load(TABLE_PATH)  # type: np.ndarray
    return pd.DataFrame(table, index=words, columns=words)


def compute_possibilities_table(words: List[str]) -> np.ndarray:
    num_words = len(words)
    # table = np.empty(shape=(num_words, num_words), dtype='uint16')
    table = np.empty(shape=(num_words, num_words), dtype='uint8')

    def f_eval_guess(guess_i: int, answer_i: int) -> int:
        """Return an integer"""
        guess = words[guess_i]
        answer = words[answer_i]
        rval = UNSAFE_eval_guess(guess=guess, answer=answer)
        # the numbers are guaranteed to be 0, 1, 2
        return array_to_integer(rval)

    word_range_1 = np.arange(num_words)
    word_range_2 = np.arange(num_words)
    combos = itertools.product(word_range_1, word_range_2)
    for guess_i, answer_i in tqdm(combos):
        table[guess_i, answer_i] = f_eval_guess(guess_i, answer_i)

    return table


if __name__ == "__main__":
    words = read_parsed_words()
    print("computing possibilities...")
    table = compute_possibilities_table(words)
    np.save('data-parsed/possibilities-table-base-3.npy', table)

