"""
Attempt to play Wordle
"""

from tkinter import RIGHT
from parse_data import read_parsed_answers, read_parsed_words
from datetime import datetime
from typing import List
import numpy as np


RIGHT_PLACE = 2
WRONG_PLACE = 1
LETTER_ABSENT = 0


def get_todays_answer() -> str:
    answers = read_parsed_answers()
    now = datetime.now()
    today = datetime(now.year, now.month, now.day)
    return answers[answers["date"] == today]["answer"].tolist()[0]


def eval_guess(guess: str, answer: str) -> List[int]:
    """Return a list saying which letters are in the right place and which are not"""
    assert len(guess) == 5
    guess = guess.lower()
    assert len(answer) == 5
    answer = answer.lower()
    return UNSAFE_eval_guess(guess, answer)


def UNSAFE_eval_guess(guess: str, answer: str) -> List[int]:
    """
    NOTE: same as eval_guess but without any checks on the input
    Assume that both guess and answer are lowercase
    """
    # pre-initialize to avoid appending
    scores = [0] * 5
    # score = 0
    for i, letter in enumerate(guess):
        if letter == answer[i]:
            scores[i] = RIGHT_PLACE
            # score += (4 ** i) * RIGHT_PLACE
        elif letter in answer:
            scores[i] = WRONG_PLACE
            # score += (4 ** i) * WRONG_PLACE
        else:
            scores[i] = LETTER_ABSENT
            # score += (4 ** i) * LETTER_ABSENT
    # return score
    return scores


# def eval_guess_fast(guess: np.array, answer: np.array) -> np.array:
#     # assert isinstance(guess, np.array)
#     # assert isinstance(answer, np.array)
#     scores = []
#     for i, letter in enumerate(np.nditer(guess)):
#         if letter == answer[i]:
#             scores.append(RIGHT_PLACE)
#         elif letter in answer:
#             scores.append(WRONG_PLACE)
#         else:
#             scores.append(LETTER_ABSENT)
#     return scores


if __name__ == "__main__":
    try:
        user_guess = ""
        num_tries = 0
        while len(user_guess) != 5:
            if num_tries > 0:
                print("Guesses must be 5 characters long")
            user_guess = input("Please enter a 5-letter guess (ctrl-C to quit): ")
            num_tries += 1

        todays_answer = get_todays_answer()
        eval = eval_guess(user_guess, todays_answer)

        print(f"guess: {user_guess}")
        print(f"answer: {todays_answer}")
        print(f"eval: {eval}")
    except KeyboardInterrupt:
        print("")
        print("ok see you")
