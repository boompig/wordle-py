import json
import logging
import os.path
import random
from typing import List, Optional, Tuple

import coloredlogs
import pandas as pd
import numpy as np
from tqdm import tqdm

from parse_data import read_all_answers, read_parsed_words, read_past_answers
from play import RIGHT_PLACE, eval_guess, WRONG_PLACE, LETTER_ABSENT
from possibilities_table import (
    array_to_integer,
    load_possibilities_table,
    load_possibilities_table_df,
)

FIRST_GUESS_WORD = "serai"


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


def get_next_guess_mean_partition(table: pd.DataFrame) -> str:
    def get_mean_partition(row) -> int:
        d = row.value_counts().to_dict()
        arr = np.array([v for v in d.values()])
        return np.mean(arr)

    part_series = table.apply(get_mean_partition, axis=1)
    part_df = pd.DataFrame(part_series, columns=["mean_partition"], index=table.index)
    i = part_df["mean_partition"].idxmin()
    return i


def get_next_guess_worst_partition(table: pd.DataFrame) -> str:
    """The table will only contain those words that remain"""
    # compute the max partitions

    def get_worst_partition(row) -> int:
        d = row.value_counts().to_dict()
        return max(d.values())

    # for each remaining guess, compute the worst partition
    part_series = table.apply(get_worst_partition, axis=1)
    # re-index it with words so return value is easier
    part_df = pd.DataFrame(part_series, columns=["worst_partition"], index=table.index)
    # return the word with the smallest worst partition
    i = part_df["worst_partition"].idxmin()
    return i


def get_next_guess(table: pd.DataFrame, strategy: str) -> str:
    if strategy == "mean_partition":
        return get_next_guess_mean_partition(table)
    elif strategy == "worst_partition":
        return get_next_guess_worst_partition(table)
    else:
        raise NotImplementedError(strategy)


def solver(
    answer: str,
    words: List[str],
    first_word: str,
    strategy: str,
    matrix_df_path: Optional[str] = None,
    verbose: Optional[bool] = True,
) -> Tuple[bool, int, List[str]]:
    """
    :param first_word: The first word to use
    :param verbose: Control whether we are actually outputing or not
    The method *must not* modify the table.
    """

    def solver_print(text: str):
        if verbose:
            print(text)

    if matrix_df_path and matrix_df_path.endswith(".npy"):
        # NOTE: this doesn't use the path, just loads the standard npy file for speed
        table = load_possibilities_table(words)
    else:
        table = load_possibilities_table_df(matrix_df_path)

    guesses = []  # type: List[str]
    guess = first_word
    is_solved = False

    while len(guesses) < 6 and not is_solved:
        if guesses == []:
            guess = first_word
        else:
            guess = get_next_guess(table, strategy=strategy)

        guesses.append(guess)

        solver_print(f"{len(guesses)}. Guessed {guess}")
        guess_result = eval_guess(guess, answer)
        solver_print(f"Guess result: {guess_result}")

        if guess_result == [
            RIGHT_PLACE,
            RIGHT_PLACE,
            RIGHT_PLACE,
            RIGHT_PLACE,
            RIGHT_PLACE,
        ]:
            is_solved = True
            break
        else:
            table = prune_table(table, guess, guess_result)
            solver_print(f"There are now {table.shape[0]} possibilities")

    if is_solved:
        solver_print(
            f"You solved it after {len(guesses)} guesses! The word was {answer}"
        )
    else:
        solver_print(f"You failed to guess the word. The word was {answer}")
    return is_solved, len(guesses), guesses


def eval_solver(
    words: List[str],
    num_answers: int,
    first_word: str,
    strategy: str,
    out_dir: str,
    matrix_df_path: Optional[str] = None,
):
    """
    Evaluate the solver on the first `num_answers` past answers
    If `num_answers` is more than the number of past answers, will display a warning and will instead use *all* answers - past and future
    """
    if not os.path.exists(out_dir):
        logging.critical("out_dir %s does not exist", out_dir)
        exit(1)

    # NOTE to self: for the future blog post, it took about 5 minutes to run this for all answers
    possible_answers = read_past_answers()
    # we use this variable as part of the filename
    dataset = "past-answers"
    if num_answers >= 0:
        print(f"Limiting testing to first {num_answers} answers")
        possible_answers = possible_answers[:num_answers]
    if num_answers > len(possible_answers):
        logging.warning(
            "Since num_answers is more than the number of past answers, going to use answers from the future"
        )
        possible_answers = read_all_answers()
        # change the dataset
        dataset = "future-answers"
    if num_answers > len(possible_answers):
        logging.warning(
            "num_answers (%d) is more than the number of answers available (%d), using all answers",
            num_answers,
            len(possible_answers),
        )
        num_answers = len(possible_answers)

    d = {}
    print(
        f"Solving {len(possible_answers)} puzzles with first word {first_word} and strategy {strategy}..."
    )
    for answer in tqdm(possible_answers):
        is_solved, num_guesses, guesses = solver(
            answer,
            words,
            first_word=first_word,
            strategy=strategy,
            matrix_df_path=matrix_df_path,
            verbose=False,
        )
        d[answer] = {
            "is_solved": is_solved,
            "num_guesses": num_guesses,
            "guesses": guesses,
        }
        if not is_solved:
            logging.error(f"failed to solve when answer was {answer}")

    if matrix_df_path:
        # we want to record that we used a custom matrix here
        out_fname = f"data-parsed/solver-eval/solver-eval-strat-{strategy}-{dataset}-{len(possible_answers)}-{first_word}-custom-matrix.json"
    else:
        out_fname = f"data-parsed/solver-eval/solver-eval-strat-{strategy}-{dataset}-{len(possible_answers)}-{first_word}.json"
    out_path = os.path.join(out_dir, out_fname)

    out = {
        "per_word_results": d,
        "first_word": first_word,
        "num_answers_tested": len(possible_answers),
        "strategy": strategy,
        "dataset": dataset,
    }
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=4, sort_keys=True)
    print(f"Eval done. Wrote to {out_path}")
    rows = []
    for answer, v in d.items():
        rows.append(
            {
                "answer": answer,
                "num_guesses": v["num_guesses"],
                "is_solved": v["is_solved"],
            }
        )
    df = pd.DataFrame(rows)
    print(f"Mean # of guesses per puzzle: {df.num_guesses.mean():.2f}")
    num_unsolved = df[df.is_solved == False].shape[0]
    num_solved = df[df.is_solved].shape[0]
    num_puzzles = len(df)
    print(f"# puzzles solved: {num_solved} ({num_solved / num_puzzles * 100:.1f}%)")
    print(
        f"# puzzles unsolved: {num_unsolved} ({num_unsolved / num_puzzles * 100:.1f})%"
    )


def get_interactive_guess_result(guess: str) -> List[int]:
    valid_vals = [LETTER_ABSENT, WRONG_PLACE, RIGHT_PLACE]

    while True:
        print("")
        print(f"Please enter the wordle result for the guess {guess}.")
        print("Enter 5 numbers with spaces between them.")
        print(
            f"Use {LETTER_ABSENT} if the letter isn't present, {WRONG_PLACE} if the letter is present but in the wrong place, and {RIGHT_PLACE} if the letter is in the right place."
        )
        uin = input("> ")
        items = uin.strip().split(" ")
        if len(items) == 5 and all([item.isdigit() for item in items]):
            arr = [int(item) for item in items]
            if all([v in valid_vals for v in arr]):
                return arr


def play_with_solver(
    words: List[str],
    first_word: str,
    strategy: str,
    matrix_df_path: Optional[str] = None,
) -> Tuple[bool, int, List[str]]:
    """Play interactively with the solver when you don't know the answer"""

    # table = load_possibilities_table(words)
    table = load_possibilities_table_df(matrix_df_path)
    guesses = []  # type: List[str]
    guess = first_word
    is_solved = False

    while len(guesses) < 6 and not is_solved:
        if guesses == []:
            guess = first_word
        else:
            guess = get_next_guess(table, strategy)

        guesses.append(guess)

        print(f"{len(guesses)}. Guessed {guess}")
        guess_result = get_interactive_guess_result(guess)
        # print(f"Guess result: {guess_result}")

        if guess_result == [
            RIGHT_PLACE,
            RIGHT_PLACE,
            RIGHT_PLACE,
            RIGHT_PLACE,
            RIGHT_PLACE,
        ]:
            is_solved = True
            break
        else:
            table = prune_table(table, guess, guess_result)
            print(f"There are now {table.shape[0]} possibilities")

    if is_solved:
        print(f"We solved it after {len(guesses)} guesses! The word was {guesses[-1]}.")
    else:
        print(f"Failed to guess the word after {len(guesses)}.")
    return is_solved, len(guesses), guesses


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="Can specify a random seed to control randomness when choosing a word",
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        choices=["play", "eval_solver", "interactive"],
        help="""What do you want to do?
play -> have the solver solve a random puzzle
eval_solver -> evaluate the solver on all the past answers and write stats out to a file
interactive -> have the solver help you solve a puzzle with an unknown answer interactively""",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-answers",
        type=int,
        help="At most how many answers to process (for eval_solver). -1 means process up to today.",
        default=-1,
    )
    parser.add_argument(
        "-f",
        "--first-word",
        type=str,
        help="The first word to guess",
        default=FIRST_GUESS_WORD,
    )
    parser.add_argument(
        "-t",
        "--strategy",
        choices=["mean_partition", "worst_partition"],
        type=str,
        default="worst_partition",
        help="The strategy to use when selecting the next guess",
    )
    parser.add_argument(
        "-m",
        "--matrix-path",
        type=str,
        help="If specified, use this path to the matrix dataframe instead of the default",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory where eval_solver will write files (must exist)",
        default="data-parsed/solver-eval",
    )
    args = parser.parse_args()
    coloredlogs.install()

    if args.seed >= 0:
        random.seed(args.seed)

    words = read_parsed_words()

    # check the input word
    if len(args.first_word) != 5:
        print("ERROR: first word must be 5 characters long")
        exit(1)
    elif args.first_word not in words:
        print("ERROR: first word a valid 5-letter word")
        exit(1)

    if args.action == "play":
        answer = random.choice(words)
        print(f"Chose random word for answer: {answer}")
        solver(
            answer,
            words,
            first_word=args.first_word,
            strategy=args.strategy,
            matrix_df_path=args.matrix_path,
        )
    elif args.action == "eval_solver":
        eval_solver(
            words,
            num_answers=args.num_answers,
            first_word=args.first_word,
            strategy=args.strategy,
            matrix_df_path=args.matrix_path,
            out_dir=args.output_dir,
        )
    elif args.action == "interactive":
        # answer = random.choice(words)
        # print(f"Chose random word for answer: {answer}")
        play_with_solver(
            words,
            first_word=args.first_word,
            strategy=args.strategy,
            matrix_df_path=args.matrix_path,
        )
    else:
        raise NotImplementedError
