"""
This file contains all the code to play Wordle
"""

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


def get_user_guess(words: List[str]) -> str:
    num_tries = 0
    user_guess = ""
    while len(user_guess) != 5 or user_guess not in words:
        if num_tries > 0:
            if len(user_guess) != 5:
                print("Guesses must be 5 characters long")
            elif user_guess not in words:
                print("That is not a valid word")
        user_guess = input("Please enter a 5-letter guess (ctrl-C to quit): ")
        user_guess = user_guess.lower()
        num_tries += 1
    return user_guess


if __name__ == "__main__":
    words = read_parsed_words()
    guesses = []  # type: List[str]
    is_solved = False
    answer = get_todays_answer().lower()
    try:
        while len(guesses) < 6 and not is_solved:
            guess = get_user_guess(words)
            guesses.append(guess)
            guess_result = eval_guess(guess, answer)

            print(f"Guess #{len(guesses)}")
            print(f"You guessed: {guess}")
            print(f"Guess result: {guess_result}")
            if guess == answer:
                is_solved = True

        if is_solved:
            print(f"You solved it after {len(guesses)} guesses! The word was {answer}")
        else:
            print(f"You failed to guess the word. The word was {answer}")
    except KeyboardInterrupt:
        print("")
        print("ok see you")
