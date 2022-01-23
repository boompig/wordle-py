import json
import logging
import random
from typing import List, Optional, Tuple

import coloredlogs
import pandas as pd
from tqdm import tqdm

from parse_data import read_parsed_words, read_past_answers
from play import RIGHT_PLACE, eval_guess, WRONG_PLACE, LETTER_ABSENT
from possibilities_table import array_to_integer, load_possibilities_table

FIRST_GUESS_WORD = "serai"
# FIRST_GUESS_WORD = "tares"


def prune_table(table: pd.DataFrame, last_guess: str, guess_result: List[int]):
    # modify the table
    columns_to_keep = []
    rval = array_to_integer(guess_result)
    for answer in table.columns:
        if table.loc[last_guess][answer] == rval:
            columns_to_keep.append(answer)
    # print("We're keeping the following columns:")
    # print(columns_to_keep)
    table = table[columns_to_keep]
    table = table[table.index.isin(columns_to_keep)]
    return table

def get_next_guess(table: pd.DataFrame):
    """The table will only contain those words that remain"""
    # compute the max partitions

    def get_worst_partition(row) -> int:
        d = row.value_counts().to_dict()
        return max(d.values())

    # for each remaining guess, compute the worst partition
    part_series = table.apply(get_worst_partition, axis=1)
    # re-index it with words so return value is easier
    part_df = pd.DataFrame(
        part_series,
        columns=['worst_partition'],
        index=table.index
    )
    # return the word with the smallest worst partition
    i = part_df['worst_partition'].idxmin()
    # print(part_df)
    # print(part_df.loc[i])
    # print(i)
    return i


def solver(answer: str, words: List[str], verbose: Optional[bool] = True) -> Tuple[bool, int, List[str]]:
    """
    :param verbose: Control whether we are actually outputing or not
    :param table: Optionally supply the possibilities table.
    The method *must not* modify the table.
    """
    def solver_print(text: str):
        if verbose:
            print(text)

    table = load_possibilities_table(words)

    guesses = []  # type: List[str]
    guess = FIRST_GUESS_WORD
    is_solved = False

    while len(guesses) < 6 and not is_solved:
        if guesses == []:
            guess = FIRST_GUESS_WORD
        else:
            guess = get_next_guess(table)

        guesses.append(guess)

        solver_print(f"{len(guesses)}. Guessed {guess}")
        guess_result = eval_guess(guess, answer)
        solver_print(f"Guess result: {guess_result}")

        if guess_result == [RIGHT_PLACE, RIGHT_PLACE, RIGHT_PLACE, RIGHT_PLACE, RIGHT_PLACE]:
            is_solved = True
            break
        else:
            table = prune_table(table, guess, guess_result)
            # print(table)
            solver_print(f"There are now {table.shape[0]} possibilities")

    if is_solved:
        solver_print(f"You solved it after {len(guesses)} guesses! The word was {answer}")
    else:
        solver_print(f"You failed to guess the word. The word was {answer}")
    return is_solved, len(guesses), guesses


def eval_solver(words: List[str]):
    # NOTE to self: for the future blog post, it took about 5 minutes to run this for all answers
    possible_answers = read_past_answers()

    d = {}
    for answer in tqdm(possible_answers):
        is_solved, num_guesses, guesses = solver(answer, words, verbose=False)
        d[answer] = {
            "is_solved": is_solved,
            "num_guesses": num_guesses,
            "guesses": guesses,
        }
        if not is_solved:
            logging.error(f"failed to solve when answer was {answer}")

    out_fname = "data-parsed/solver-eval-past-answers.json"
    with open(out_fname, "w") as fp:
        json.dump(d, fp, indent=4, sort_keys=True)
    print("Eval done.")
    rows = []
    for answer, v in d.items():
        rows.append({
            "answer": answer,
            "num_guesses": v["num_guesses"],
            "is_solved": v["is_solved"],
        })
    df = pd.DataFrame(rows)
    print(f"Mean # of guesses per puzzle: {df.num_guesses.mean()}")
    num_unsolved = df[df.is_solved == False].shape[0]
    print(f"# puzzles unsolved: {num_unsolved}")
    # from pprint import pprint
    # pprint(d)


def get_interactive_guess_result(guess: str) -> List[int]:
    valid_vals = [LETTER_ABSENT, WRONG_PLACE, RIGHT_PLACE]

    while True:
        print("")
        print(f"Please enter the wordle result for the guess {guess}.")
        print("Enter 5 numbers with spaces between them.")
        print(f"Use {LETTER_ABSENT} if the letter isn't present, {WRONG_PLACE} if the letter is present but in the wrong place, and {RIGHT_PLACE} if the letter is in the right place.")
        uin = input("> ")
        items = uin.strip().split(" ")
        if len(items) == 5 and all([item.isdigit() for item in items]):
            arr = [int(item) for item in items]
            if all([v in valid_vals for v in arr]):
                return arr


def play_with_solver(words: List[str]):
    """Play interactively with the solver when you don't know the answer"""

    table = load_possibilities_table(words)
    guesses = []  # type: List[str]
    guess = FIRST_GUESS_WORD
    is_solved = False

    while len(guesses) < 6 and not is_solved:
        if guesses == []:
            guess = FIRST_GUESS_WORD
        else:
            guess = get_next_guess(table)

        guesses.append(guess)

        print(f"{len(guesses)}. Guessed {guess}")
        guess_result = get_interactive_guess_result(guess)
        # print(f"Guess result: {guess_result}")

        if guess_result == [RIGHT_PLACE, RIGHT_PLACE, RIGHT_PLACE, RIGHT_PLACE, RIGHT_PLACE]:
            is_solved = True
            break
        else:
            table = prune_table(table, guess, guess_result)
            # print(table)
            print(f"There are now {table.shape[0]} possibilities")

    if is_solved:
        print(f"We solved it after {len(guesses)} guesses! The word was {guesses[-1]}.")
    else:
        print(f"Failed to guess the word after {len(guesses)}.")
    return is_solved, len(guesses), guesses


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=-1,
                        help="Can specify a random seed to control randomness when choosing a word")
    parser.add_argument("-a", "--action", type=str, choices=["play", "eval_solver", "interactive"],
                        help="What do you want to do?", required=True)
    args = parser.parse_args()
    coloredlogs.install()

    if args.seed >= 0:
        random.seed(args.seed)

    words = read_parsed_words()

    if args.action == "play":
        answer = random.choice(words)
        print(f"Chose random word for answer: {answer}")
        solver(answer, words)
    elif args.action == "eval_solver":
        eval_solver(words)
    elif args.action == "interactive":
        # answer = random.choice(words)
        # print(f"Chose random word for answer: {answer}")
        play_with_solver(words)
    else:
        raise NotImplementedError

