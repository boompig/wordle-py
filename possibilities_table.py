"""
This file will build the possibilities matrix
"""

import itertools
import os.path
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TABLE_PATH = os.path.join(BASE_DIR, "data-parsed/possibilities-table-base-3.npy")
TABLE_DF_PATH = os.path.join(
    BASE_DIR, "data-parsed/possibilities-table-base-3.parquet.gzip"
)
TABLE_PATH_ASYMMETRIC = os.path.join(
    BASE_DIR, "data-parsed/possibilities-table-asymmetric-base-3.npy"
)
TABLE_PATH_CHEATING = "./data-parsed/possibilities-table-cheating-base-3.npy"


from parse_data import read_all_answers, read_parsed_words
from play import LETTER_ABSENT, RIGHT_PLACE, WRONG_PLACE, UNSAFE_eval_guess


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


def guess_response_from_string(guess_response: str) -> int:
    assert len(guess_response) == 5

    def char_to_base_3(s: str) -> int:
        if s == "G":
            return RIGHT_PLACE
        elif s == "Y":
            return WRONG_PLACE
        elif s == "B":
            return LETTER_ABSENT
        else:
            raise Exception(s)

    arr = [char_to_base_3(c) for c in guess_response]
    return array_to_integer(arr)


def guess_response_to_string(rval: int) -> str:
    def base_3_to_char(val: int) -> str:
        if val == RIGHT_PLACE:
            return "G"
        elif val == WRONG_PLACE:
            return "Y"
        elif val == LETTER_ABSENT:
            return "B"
        else:
            raise Exception(val)

    arr = integer_to_arr(rval)
    chars = map(base_3_to_char, arr)
    return "".join(chars)


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


def load_possibilities_table(words: List[str]) -> pd.DataFrame:
    """
    Return a dataframe
    The index will represent guesses
    The columns will represent answers
    """
    table = np.load(TABLE_PATH)  # type: np.ndarray
    return pd.DataFrame(table, index=words, columns=words)


def load_possibilities_table_df(path: Optional[str] = None) -> pd.DataFrame:
    """Same as above but will load the dataframe directly
    :param path: May optionally specify a path
    """
    if path is None:
        path = TABLE_DF_PATH
    return pd.read_parquet(path)


def compute_possibilities_table(words: List[str]) -> np.ndarray:
    num_words = len(words)
    print(f"computing {num_words}x{num_words} possibilities matrix...")
    table = np.empty(shape=(num_words, num_words), dtype="uint8")

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


def compute_possibilities_table_asymmetric(
    guesses: List[str], answers: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    num_guesses = len(guesses)
    num_answers = len(answers)
    print(f"computing {num_guesses}x{num_answers} possibilities matrix...")
    table = np.empty(shape=(num_guesses, num_answers), dtype="uint8")

    # we have to do this in a clever way, where guesses[i] == answers[i] for each i < len(answers)
    answers.sort()
    remaining_words = list(set(guesses) - set(answers))
    # we want a stable order for these
    remaining_words.sort()
    guesses = answers + remaining_words
    # check that we've achieved our goal
    for i in range(len(answers)):
        assert guesses[i] == answers[i]

    def f_eval_guess(guess_i: int, answer_i: int) -> int:
        """Return an integer"""
        guess = guesses[guess_i]
        answer = answers[answer_i]
        rval = UNSAFE_eval_guess(guess=guess, answer=answer)
        # the numbers are guaranteed to be 0, 1, 2
        return array_to_integer(rval)

    guess_range = np.arange(num_guesses)
    answer_range = np.arange(num_answers)
    combos = itertools.product(guess_range, answer_range)
    for guess_i, answer_i in tqdm(combos):
        table[guess_i, answer_i] = f_eval_guess(guess_i, answer_i)

    return table, guesses, answers


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        required=True,
        choices=["full", "asymmetric", "cheating"],
        default="full",
        help="What kind of matrix to generate",
    )
    args = parser.parse_args()

    if args.type == "full":
        words = read_parsed_words()
        print("computing possibilities...")
        table = compute_possibilities_table(words)
        np.save(TABLE_PATH, table)
    elif args.type == "asymmetric":
        words = read_parsed_words()
        answers = read_all_answers()
        print("computing possibilities...")
        table, row_keys, column_keys = compute_possibilities_table_asymmetric(
            words, answers
        )
        np.save(TABLE_PATH_ASYMMETRIC, table)
        with open("data-parsed/possibilities-keys-asymmetric.pickle", "wb") as fp:
            pickle.dump((row_keys, column_keys), fp)
    elif args.type == "cheating":
        answers = read_all_answers()
        print("computing possibilities...")
        table = compute_possibilities_table(answers)
        np.save(TABLE_PATH_CHEATING, table)
