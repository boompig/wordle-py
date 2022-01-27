"""
Verifies that all generated decision trees are valid
"""

import glob
import json
import logging
import pickle

import numpy as np
import pandas as pd

from parse_data import read_all_answers, read_parsed_words
from play import eval_guess
from possibilities_table import array_to_integer, guess_response_to_string


def find_answer_in_tree(
    answer: str,
    tree: dict,
    depth: int,
    guess_words: list[str],
    answer_words: list[str],
    guesses: list[str] | None = None,
) -> tuple[int, list[str]]:
    """
    Find the given answer in the provided tree
    Return the depth
    Throw an error if it is not found
    """

    if guesses is None:
        guesses = []

    # get the top-level key
    root_words = list(tree.keys())
    assert len(root_words) == 1
    root_word = root_words[0]  # type: int
    guess_word = guess_words[int(root_word)]
    logging.info("depth %d, guessing %s", depth, guess_word)
    guesses.append(guess_word)

    rv = eval_guess(guess_word, answer)
    rvi = array_to_integer(rv)
    rvs = guess_response_to_string(rvi)
    logging.info("Response: %s", rvs)

    if rvi == 242:
        logging.info("Correct!")
        return depth, guesses

    action_map = tree[root_word]
    if str(rvi) in action_map:
        return find_answer_in_tree(
            answer=answer,
            tree=action_map[str(rvi)],
            depth=depth + 1,
            guess_words=guess_words,
            answer_words=answer_words,
            guesses=guesses,
        )
    else:
        raise Exception(
            "Failed to find answer %s. No action to perform here: %s. Path is %s"
            % (answer, rvs, str(guesses))
        )


def check_file(path: str, dictionary: str) -> pd.DataFrame:
    tree = {}
    root_word = path.split("/")[-1].split(".")[0]
    with open(path) as fp:
        tree = json.load(fp)

    guess_words = []
    answer_words = []

    if dictionary == "answers":
        guess_words = read_all_answers()
        answer_words = read_all_answers()
    elif dictionary == "asymmetric":
        # actually need to use another set of files
        with open("./data-parsed/possibilities-keys-asymmetric.pickle", "rb") as fp:
            guess_words, answer_words = pickle.load(fp)

        # now verify that they are loaded correctly
        for i in range(len(answer_words)):
            assert answer_words[i] == guess_words[i]
    else:
        raise Exception(dictionary)

    d = {}
    for answer in answer_words:
        depth, _ = find_answer_in_tree(
            answer=answer,
            tree=tree,
            depth=1,
            guess_words=guess_words,
            answer_words=answer_words,
        )
        d[answer] = depth

    df = pd.DataFrame({"answer": d.keys(), "depth": d.values()})
    # print(df)
    mean_depth = np.mean(df["depth"])
    print(f"mean depth for tree rooted at {root_word} is {mean_depth:.2f}")
    return df


def convert_tree_to_human_readable(
    tree: dict, guess_words: list[str], answer_words: list[str]
) -> dict:
    """
    Convert a decision tree to one that is human-readable
    """
    hr_tree = {}
    hr_action_map = {}

    root_words = list(tree.keys())
    assert len(root_words) == 1
    root_word = root_words[0]  # type: int
    guess_word = guess_words[int(root_word)]
    hr_tree[guess_word] = hr_action_map

    for rvi in tree[root_word]:
        rvi = int(rvi)
        rvs = guess_response_to_string(rvi)
        hr_action_map[rvs] = convert_tree_to_human_readable(
            tree=tree[root_word][str(rvi)],
            guess_words=guess_words,
            answer_words=answer_words,
        )

    return hr_tree


def convert_file_to_human_readable(path: str, dictionary: str):
    tree = {}
    root_word = path.split("/")[-1].split(".")[0]
    print(f"Converting tree {path} to human readable format")
    with open(path) as fp:
        tree = json.load(fp)

    if dictionary == "answers":
        guess_words = read_all_answers()
        answer_words = read_all_answers()
    elif dictionary == "asymmetric":
        # actually need to use another set of files
        with open("./data-parsed/possibilities-keys-asymmetric.pickle", "rb") as fp:
            guess_words, answer_words = pickle.load(fp)

        # now verify that they are loaded correctly
        for i in range(len(answer_words)):
            assert answer_words[i] == guess_words[i]
    else:
        raise Exception(dictionary)

    out_tree = convert_tree_to_human_readable(tree, guess_words, answer_words)
    out_path = f"./out/decision-trees/{dictionary}/{root_word}-hr.json"
    with open(out_path, "w") as fp:
        json.dump(out_tree, fp, sort_keys=True, indent=4)
    print(f"Wrote to {out_path}")


if __name__ == "__main__":
    # silence logging in find_answer_in_tree
    logging.basicConfig(level=logging.WARNING)

    # convert_file_to_human_readable("out/decision-trees/asymmetric/serai.json", "asymmetric")

    dictionary = "asymmetric"
    files = list(glob.iglob(f"out/decision-trees/{dictionary}/*.json"))
    files.sort()

    for file in files:
        print("Checking file %s" % file)
        check_file(file, dictionary)
