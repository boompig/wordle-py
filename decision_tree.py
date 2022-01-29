import json
import logging
import os.path
import pickle
import time
from typing import Dict, Iterable, List, Set, Tuple

import coloredlogs
import numpy as np
import pandas as pd
from tqdm import tqdm

from parse_data import read_all_answers, read_parsed_words
from play import LETTER_ABSENT
from possibilities_table import (
    TABLE_PATH,
    TABLE_PATH_ASYMMETRIC,
    TABLE_PATH_CHEATING,
    integer_to_arr,
    guess_response_to_string,
)


ALL_LETTERS_CORRECT = (3 ** 5) - 1
# maximum number of guesses
# by default 6 but can be helpful to lower it to look for smaller decision trees
MAX_DEPTH = 6

# set this to true to enable debug printing
IS_DEBUG = True

# set this to logging.debug to hide it
PROGRESS_LOG_LEVEL = logging.INFO
# at what level to print. 3 is very verbose and 1 is too infrequent
MAX_PROGRESS_DEPTH = 2

# whether to use tqdm (progress bar) at a low depth
# this will mess up the output if debugging is enabled
# and obviously will slow down the solver a little
USE_TQDM_LOW_DEPTHS = True
# at which depth to enable the progress bar
# I do not recommend setting lower than 1 or 2
TQDM_DEPTH = 1

# at what level to print debug messages from Optimization #3
# logging.DEBUG hides the messages
OPT_3_LOG_LEVEL = logging.DEBUG

# whether to use optimization #4
# doesn't always make sense to turn this on
# if we use a smaller dictionary, the solver is much faster without it
# this makes the solver use a better heuristic at shallower depths to better direct the search
USE_OPT_4 = True
# at what level to output these messages
# note that logging.DEBUG will *hide* the messages. This is on purpose.
OPT_4_LOG_LEVEL = logging.DEBUG

# whether to time how quickly we're solving stuff
IS_TIMING_ENABLED = False
# at what depth to time
TIMING_DEPTH = 2

SORTED_GUESSES = []  # type: List[int]
GUESS_WORDS = []  # type: List[str]
ANSWER_WORDS = []  # type: List[str]

# every time we have a solution for a subtree at depth n, save the entire tree
USE_CHECKPOINTS = False
CHECKPOINT_DEPTH = 2

# if we don't care about optimality and want to just find some decision tree
# then we can just exit when we find a solution
EXIT_ON_FIRST_SOLUTION = True


def get_black_letters(
    guesses: List[int], results: List[int], guess_words: List[str]
) -> Set[str]:
    """
    Once a letter is turned black from a guess, that letter cannot be used in any subsequent word.
    Return all the unusable letters
    NOTE: This is the only method that relies on words being strings rather than integers
    NOTE: This method is on the hot path so long as pick_next_guesses_it is on the hot path
    """
    black_letters = set([])
    for (guess, result) in zip(guesses, results):
        result_arr = integer_to_arr(result)
        word = guess_words[guess]
        for (letter, val) in zip(word, result_arr):
            if val == LETTER_ABSENT:
                black_letters.add(letter)
    return black_letters


def get_human_readable_path(
    guesses: List[int], guess_results: List[int], depth: int, guess_words: List[str]
) -> str:
    """
    Used for creating file paths
    """
    path = []
    for i in range(depth):
        guess = guesses[i]
        guess_word = guess_words[guess]
        path.append(guess_word)
        if i < len(guess_results):
            rv = guess_results[i]
            path.append(guess_response_to_string(rv))
    return "_".join(path)


def get_chain(
    prev_guesses: List[int], prev_guess_results: List[int], depth: int
) -> str:
    """
    Used for debugging.
    Print the entire chain of the decision tree so far.
    NOTE: accesses GUESS_WORDS in global scope
    """
    chain = []
    for i in range(depth):
        guess = prev_guesses[i]
        guess_word = GUESS_WORDS[guess]
        chain.append(guess_word)
        if i < len(prev_guess_results):
            rv = prev_guess_results[i]
            chain.append(guess_response_to_string(rv))
    s = f"depth = {depth}: path = " + " -> ".join(chain)
    return s


def print_chain(
    prev_guesses: List[int], prev_guess_results: List[int], depth: int
) -> None:
    """
    Used for debugging.
    Print the entire chain of the decision tree so far.
    """
    s = get_chain(prev_guesses, prev_guess_results, depth)
    print(s)


def get_mean_partition(row: np.ndarray) -> float:
    _, counts = np.unique(row, return_counts=True)
    return np.mean(counts)


def get_worst_partition_bincount(row: np.ndarray) -> float:
    """An optimized version of get_worst_partition"""
    return np.bincount(row).max()


def get_mean_partition_arr(table: np.ndarray, possible_answers: Set[int]) -> np.ndarray:
    # select the columns of the table with the possible answers
    possible_answers_arr = np.array(list(possible_answers))
    m = table[:, possible_answers_arr]
    # compute the mean partition size for each row
    return np.apply_along_axis(get_mean_partition, axis=1, arr=m)


def get_worst_partition_arr(
    table: np.ndarray, possible_answers: Set[int]
) -> np.ndarray:
    # select the columns of the table with the possible answers
    possible_answers_arr = np.array(list(possible_answers))
    m = table[:, possible_answers_arr]
    # compute the mean partition size for each row
    return np.apply_along_axis(get_worst_partition_bincount, axis=1, arr=m)


def NEW_pick_next_guesses_it(
    guesses: List[int],
    guess_results: List[int],
    table: np.ndarray,
    possible_answers: Set[int],
) -> Iterable[int]:
    """Return an iterator over possible next guesses in order of our heuristic
    This method is about 40 times slower than `pick_next_guesses_it`
    Hopefully it returns values that are 40x better
    """
    possible_answers_arr = np.array(list(possible_answers))
    # select the columns of the table with the possible answers
    m = table[:, possible_answers_arr]

    # compute the largest partition size for each row
    # sort_arr = np.apply_along_axis(get_worst_partition_bincount, axis=1, arr=m)
    # compute the mean partition size for each row
    sort_arr = np.apply_along_axis(get_mean_partition, axis=1, arr=m)

    si = np.argsort(sort_arr)

    # assume that all guesses are valid for now
    valid_guesses = np.arange(table.shape[0])

    # use si to sort the valid guesses in order of the heuristic
    # smallest mean partition goes first
    sg = np.take_along_axis(valid_guesses, si, axis=0)

    for next_guess in sg:
        if next_guess in guesses:
            continue

        # logging.info(f"{next_guess} {words[next_guess]} {mean_partitions[next_guess]}")
        yield next_guess


def pick_next_guesses_it(
    guesses: List[int],
    guess_results: List[int],
    sorted_guesses: List[int],
    guess_words: List[str],
) -> Iterable[int]:
    """Return an iterator over possible next guesses.
    They are returned in the order that they are probably best.
    Guesses that contain known non-existant letters are not returned.

    NOTE: This is on the hot path. This method will be called hundreds of thousands, if not millions, of times.
    NOTE: This method is unsound - it may not return some valid guesses, leading to suboptimal results
    """
    if not EXIT_ON_FIRST_SOLUTION:
        raise Exception("Error: Using unsound method pick_next_guesses_it when trying to find optimal solution")

    black_letters = get_black_letters(guesses, guess_results, guess_words)

    for next_guess in sorted_guesses:
        if next_guess in guesses:
            continue

        w = guess_words[next_guess]
        if any([letter in black_letters for letter in w]):
            continue

        yield next_guess


def NEW_find_possible_answers(
    guesses: List[int], guess_results: List[int], table: np.ndarray
) -> Set[int]:
    """
    NOTE: this is much slower than `find_possible_answers`, maybe 20x
    Do not use this
    """
    # each guess must have a corresponding result
    assert len(guesses) == len(guess_results)
    reachable = set([])

    if not guesses:
        l = np.arange(table.shape[0])
        return set(l)

    num_answers = table.shape[0]
    m = table[guesses]
    gr = np.array(guess_results)

    for i in np.arange(num_answers):
        if (m[:, i] == gr).all():
            reachable.add(i)
    return reachable


def find_possible_answers(
    guesses: List[int], guess_results: List[int], table: np.ndarray
) -> Set[int]:
    """
    Return a list of answers that are still possible given this history
    I benchmarked this and it's pretty fast
    """
    assert len(guesses) == len(guess_results)

    # to start, we can reach all words
    reachable = set(np.arange(table.shape[0]))
    for i, guess in enumerate(guesses):
        result = guess_results[i]
        reachable_from_guess = set(np.where(table[guess] == result)[0])
        # print("Can reach %d answers from guess %d" % (len(reachable_from_guess), guess))
        reachable.intersection_update(reachable_from_guess)
    # print("Can reach %d answers total" % len(reachable))
    return reachable


def print_debug(s: str) -> None:
    if IS_DEBUG:
        print(s)


def print_chain_debug(*args) -> None:
    if IS_DEBUG:
        print_chain(*args)


def checkpoint_tree(
    guesses: List[int],
    guess_results: List[int],
    depth: int,
    tree: dict,
    guess_words: List[str],
    table: np.ndarray,
) -> None:
    human_readable_path = get_human_readable_path(
        guesses, guess_results, depth, guess_words
    )

    # figure out the dictionary used
    assert guesses != []

    dictionary = "answers"
    if table.shape == (12972, 12972):
        dictionary = "full"
    elif table.shape == (12972, 2315):
        dictionary = "asymmetric"
    else:
        dictionary = "answers"

    path = (
        f"cache/tree-checkpoints/checkpoint-{dictionary}-{human_readable_path}.pickle"
    )
    with open(path, "wb") as fp:
        pickle.dump(tree, fp)
    logging.info("Checkpointed partially solved tree")


def construct_tree(
    guesses: List[int],
    guess_results: List[int],
    table: np.ndarray,
    depth: int,
    possible_answers: Set[int],
    size_cutoff: int = -1
) -> Tuple[dict, Set[int], int, int]:
    """
    Try to construct the best tree starting from an initial guess.
    *best* is defined here as a tree that reaches the maximum number of possible answers

    :param guesses:             The guesses so far, in order
    :param guess_results:       The results of the those guesses, in order. Will be one fewer than guesses
    :param depth:               The depth of the tree. Should be the same as # of guesses
    :param possible_answers:    The set of possible answers remaining
    :param size_cutoff:         If the size of the current tree gets larger than size_cutoff, just return early.
                                -1 for no size cutoff

    Return a tuple of 3 items:
        - tree ->               Map from a root word to possible results for that root word. Each action maps to another guess and so forth
                                The tree is guaranteed to have only one top-level element: the last guess
        - found_words ->        A set of all words that are reachable from this subtree.
                                Reachable means that, if this word is the answer, we can find a unique path to that word
        - tree_size ->          The number of nodes in the tree (total number of guesses to reach all words)
                                This is a measure of optimality
        - num_states_opened ->  The number of states that we tried when creating this tree
                                This is a measure of how good our heuristic / search is
                                Does not necessarily translate to a shorter search in wall-clock time
    """
    assert depth > 0
    assert len(guesses) == depth
    assert len(guess_results) == depth - 1

    latest_guess = guesses[-1]

    action_map = {}  # type: Dict[int, dict]
    tree = {}  # type: Dict[int, dict]
    tree[int(latest_guess)] = action_map
    tree_found_words = set([])
    num_states_opened = 1  # we tried the root
    tree_size = 0

    if latest_guess in possible_answers:
        # include the root if it's a viable answer
        # note that we can guess words which are not viable to narrow down possibilities
        tree_found_words.add(latest_guess)
        tree_size += depth

    if depth >= MAX_DEPTH:
        return tree, tree_found_words, tree_size, num_states_opened

    possible_results, counts = np.unique(table[latest_guess], return_counts=True)
    # a further optimization: we should try the partitions with the *most* possible answers *first*
    si = np.argsort(-1 * counts)
    possible_results = np.take_along_axis(possible_results, si, axis=0)

    if USE_TQDM_LOW_DEPTHS and depth == TQDM_DEPTH:
        pr_it = tqdm(possible_results)
    else:
        pr_it = possible_results

    if IS_TIMING_ENABLED and TIMING_DEPTH == depth:
        start = time.time()

    is_early_exit = False
    for guess_result in pr_it:
        if guess_result == ALL_LETTERS_CORRECT and latest_guess in possible_answers:
            # we've guessed the word. we're good.
            # no need to add it to the decision tree
            # tree_found_words.add(latest_guess)
            assert latest_guess in tree_found_words
            continue
        elif (
            guess_result == ALL_LETTERS_CORRECT and latest_guess not in possible_answers
        ):
            # logging.warning("We should not be looking here")
            continue

        possible_answers_for_result = np.where(table[latest_guess] == guess_result)[0]
        possible_answers_for_result_s = set(possible_answers_for_result)
        new_possible_answers = possible_answers.intersection(
            possible_answers_for_result_s
        )

        if not new_possible_answers:
            # this is a combo of guesses that simply doesn't yield any valid words remaining
            # so there's no need to add it to the decision tree
            continue

        # TODO: uses globals and violates scope
        next_guesses_it = pick_next_guesses_it(
            guesses, guess_results + [guess_result], SORTED_GUESSES, GUESS_WORDS
        )
        # has to be -1 to match size_cutoff argument
        best_subtree_size = -1
        if size_cutoff > -1:
            # we only have the budget of whatever is remaining from our top-level cutoff
            best_subtree_size = size_cutoff - tree_size
            # logging.warning("Changed best_subtree_size to %d", best_subtree_size)
        best_subtree_found_words = set([])

        # number of guesses tried to find the optimal subtree
        num_guesses_tried = 0

        # true iff we found a guess that solves this subtree (works with this guess result)
        is_subtree_solved = False

        is_opt_4_enabled = False

        if len(new_possible_answers) == 1:
            # Optimization #1: if there is only one possible answer, then we guess only that answer
            # then we guess that word
            answer = list(new_possible_answers)[0]
            next_guesses_it = [answer]
            # logging.info("Applying optimization #1 at depth %d", depth)
        elif depth == (MAX_DEPTH - 1) and len(new_possible_answers) > 1:
            # Optimization #2: if we have 1 guess remaining and there are many (>1) possible words
            # then we can just guess any of those words
            # it doesn't matter, we will only be able to reach one of them anyway
            # save time on not trying more possibilities
            logging.debug("Applying optimization #2 at depth 5 - early exit")
            is_early_exit = True
            break
        elif depth == (MAX_DEPTH - 2):
            # Optimization #3: we have only 2 guesses left
            # we need to pick the guess that divides the space such that, for all possible remaining answers, we can solve the puzzle using the last guess
            # i.e. we want all partitions to have size 1
            worst_partition_arr = get_worst_partition_arr(table, new_possible_answers)
            # is there any guess that has a worst partition of 1?
            good_guesses = np.where(worst_partition_arr == 1)[0]
            if good_guesses.size == 0:
                logging.log(
                    OPT_3_LOG_LEVEL,
                    "Optimization #3 enabled: there is *no* good partition at depth 4. Exiting early.",
                )
                is_early_exit = True
                break
                # # there are no good guesses
                # # just pick a random word
                # rw = list(new_possible_answers)[0]
                # next_guesses_it = [rw]
            else:
                # this is our optimal partition
                opt = good_guesses[0]
                next_guesses_it = [opt]
                logging.log(
                    OPT_3_LOG_LEVEL,
                    "Optimization #3 enabled: Found the optimal partition at depth 4",
                )
        elif USE_OPT_4 and depth <= (MAX_DEPTH - 3):
            # Optimization #4
            # instead of using our weak heuristic, use a slower but better heuristic to select guesses
            logging.log(OPT_4_LOG_LEVEL, "Optimization #4 enabled at depth %d", depth)
            next_guesses_it = NEW_pick_next_guesses_it(
                guesses, guess_results, table, new_possible_answers
            )
            is_opt_4_enabled = True

        for next_guess in next_guesses_it:
            # ---- this is all debug code
            if USE_OPT_4 and is_opt_4_enabled:
                logging.log(
                    OPT_4_LOG_LEVEL,
                    "[d=%d] Optimization #4 enabled. Trying guess %d instead for guess_result %d",
                    depth,
                    next_guess,
                    guess_result,
                )
            # ---- this is all debug code

            subtree, subtree_found_words, subtree_size, subtree_states_opened = construct_tree(
                guesses=guesses + [next_guess],
                guess_results=guess_results + [guess_result],
                table=table,
                possible_answers=new_possible_answers,
                depth=depth + 1,
                size_cutoff=best_subtree_size,
            )

            num_states_opened += subtree_states_opened
            num_guesses_tried += 1

            if len(subtree_found_words) == len(new_possible_answers):
                if best_subtree_size == -1 or subtree_size < best_subtree_size:
                    # need to convert numpy type into python-native type for later serialization
                    action_map[int(guess_result)] = subtree
                    # update best subtree size
                    best_subtree_found_words = subtree_found_words
                    best_subtree_size = subtree_size

                    is_subtree_solved = True
                    # ---- this is all debug code
                    if not EXIT_ON_FIRST_SOLUTION and depth <= 2:
                        path = get_chain(guesses, guess_results + [guess_result], depth)
                        logging.info("Word %s solves subtree %s with size %d. Looking for better solution.",
                                    GUESS_WORDS[next_guess], path, subtree_size)
                    # ---- this is all debug code
                else:
                    # ---- this is all debug code
                    if not EXIT_ON_FIRST_SOLUTION and depth <= 1:
                        path = get_chain(guesses, guess_results + [guess_result], depth)
                        logging.warning("Word %s solves subtree %s with size %d, but best is %d. Looking for better solution.",
                                    GUESS_WORDS[next_guess], path, subtree_size, best_subtree_size)
                    # ---- this is all debug code
                    pass

                # NOTE: if we don't care about optimality and just want to find *some* subtree that solves
                # then we can early-exit here
                if EXIT_ON_FIRST_SOLUTION:
                    break
            else:
                # subtree is not solved
                # therefore this word should not be considered as a valid guess
                pass

        tree_found_words.update(best_subtree_found_words)
        tree_size += best_subtree_size

        if size_cutoff > 0 and tree_size > size_cutoff:
            # ---- this is all debug code
            if depth <= 2:
                path = get_chain(guesses, guess_results + [guess_result], depth)
                logging.warning("Exceeded size cutoff of %d in subtree. Path: %s", size_cutoff, path)
            # ---- this is all debug code
            is_early_exit = True
            break

        # ---- this is all debug code
        if USE_OPT_4 and is_opt_4_enabled:
            logging.log(
                OPT_4_LOG_LEVEL,
                "[d=%d] Optimization #4 enabled. # guesses tried for subtree with guess_result %d: %d (is subtree solved? %d)",
                depth,
                guess_result,
                num_guesses_tried,
                is_subtree_solved,
            )
        # ---- this is all debug code

        if not is_subtree_solved:
            # early exit. don't bother trying the other guess results
            # ---- this is all debug code
            if depth <= 3:
                path = get_chain(guesses, guess_results + [guess_result], depth)
                # logging.warning("[d=%d] path %s is a dead end or suboptimal. Backtracking.", depth, path)
            # ---- this is all debug code
            is_early_exit = True
            break

    # ---- this is all debug code
    if not is_early_exit and depth <= MAX_PROGRESS_DEPTH:
        path = get_chain(guesses[:-1], guess_results, depth - 1)
        logging.log(
            PROGRESS_LOG_LEVEL,
            "Guess %s solves subtree: %s",
            GUESS_WORDS[latest_guess],
            path,
        )
    if USE_CHECKPOINTS and not is_early_exit and depth <= CHECKPOINT_DEPTH:
        checkpoint_tree(guesses, guess_results, depth, tree, GUESS_WORDS, table)
    # ---- this is all debug code

    if IS_TIMING_ENABLED and TIMING_DEPTH == depth:
        stop = time.time()
        path = get_chain(guesses, guess_results, depth)
        logging.info(
            "Expanded %d states at depth %d. Took %.2f seconds. Path: %s",
            num_states_opened,
            depth,
            stop - start,
            path,
        )

    return tree, tree_found_words, tree_size, num_states_opened


def check_is_reachable(
    guesses: List[int], guess_results: List[int], table: np.ndarray, target_word: int
) -> bool:
    logging.warning(
        "This is a debug function and should not be run when going for speed"
    )
    is_reachable = True
    print(f"Checking reachability of answer {target_word}...")
    for i, guess in enumerate(guesses):
        expected = guess_results[i]
        actual = table[guess, target_word]
        if expected == actual:
            print(f"{i + 1}. Reachable from guess {guess}")
        else:
            print(
                f"{i + 1}. Not reachable from guess {guess}. Expected {expected} ({integer_to_arr(expected)}), actual {actual} ({integer_to_arr(actual)})"
            )
            is_reachable = False
    return is_reachable


def solve(dictionary: str, first_word: str, max_depth: int, find_optimal: bool = False):
    """
    :param find_optimal:     Whether to solve the decision tree optimally or just find some solution
    """
    assert first_word is not None
    logging.info("Using dictionary '%s'", dictionary)
    logging.info("Building decision tree using root word %s", first_word)
    logging.info("Max depth is set to %d", max_depth)

    words = []  # type: List[str]
    if dictionary == "full" or dictionary == "asymmetric":
        words = read_parsed_words()
    else:
        words = [word.lower() for word in read_all_answers()]

    guess_words = words
    answer_words = words
    if dictionary == "asymmetric":
        # actually need to use another set of files
        with open("./data-parsed/possibilities-keys-asymmetric.pickle", "rb") as fp:
            guess_words, answer_words = pickle.load(fp)

        # now verify that they are loaded correctly
        for i in range(len(answer_words)):
            assert answer_words[i] == guess_words[i]

    global MAX_DEPTH
    MAX_DEPTH = max_depth
    global EXIT_ON_FIRST_SOLUTION
    EXIT_ON_FIRST_SOLUTION = not find_optimal

    if not EXIT_ON_FIRST_SOLUTION:
        logging.warning("Looking for optimal decision tree rather than the first one we find")
        logging.warning("This takes a while...")
    else:
        logging.warning("Not looking for an optimal solution, just *a* solution")

    print(f"Loaded {len(words)} words")
    # NOTE: this is bad practice but it is accessed in the global scope
    global GUESS_WORDS
    GUESS_WORDS = guess_words
    global ANSWER_WORDS
    ANSWER_WORDS = answer_words

    table = np.zeros(shape=(1, 1), dtype='uint8')  # type: np.ndarray
    if dictionary == "full":
        table = np.load(TABLE_PATH)  # type: np.ndarray
    elif dictionary == "asymmetric":
        table = np.load(TABLE_PATH_ASYMMETRIC)  # type: np.ndarray
    else:
        table = np.load(TABLE_PATH_CHEATING)  # type: np.ndarray
    print(f"Loaded {table.shape} table")

    mean_part_df = None
    cache_path = f"cache/mean_partition-{args.dictionary}.parquet"
    if not os.path.exists(cache_path):
        df = pd.DataFrame(table, index=guess_words, columns=answer_words)
        df["word_index"] = np.arange(len(guess_words))
        print("Computing mean partition...")
        mean_part_df = df.apply(get_mean_partition, axis=1)
        mean_part_df = pd.DataFrame(mean_part_df, columns=["mean_partition"])
        mean_part_df["word_index"] = np.arange(len(guess_words))
        print(mean_part_df.sort_values(by="mean_partition"))
        mean_part_df.to_parquet(cache_path)
        print(f"Saved mean partition df to file {cache_path}")
    else:
        mean_part_df = pd.read_parquet(cache_path)
        print("Loaded mean partition DF from cache")

    # lower score is better
    # sort word indexes based on their score in above score dict
    # sorted in ascending order (lowest score first)
    # NOTE: this is bad practice but it is accessed in the global scope
    global SORTED_GUESSES
    SORTED_GUESSES = mean_part_df.sort_values("mean_partition")["word_index"].values

    possible_answers = set([i for i in range(len(answer_words))])

    try:
        root_word_index = guess_words.index(first_word)
    except ValueError as e:
        logging.error("First word %s is not in the list of guess words", first_word)
        raise e

    # do we have any solved subtrees saved for this starting word?
    # tree = load_tree(dictionary, first_word)

    # set optimization options based on the dictionary
    if dictionary == "answers":
        global IS_DEBUG
        IS_DEBUG = False
        global PROGRESS_LOG_LEVEL
        PROGRESS_LOG_LEVEL = logging.DEBUG
        global USE_TQDM_LOW_DEPTHS
        USE_TQDM_LOW_DEPTHS = False
        if MAX_DEPTH == 6 and EXIT_ON_FIRST_SOLUTION:
            global USE_OPT_4
            USE_OPT_4 = False
            logging.info("OPT_4 is disabled")
            pass
        global IS_TIMING_ENABLED
        IS_TIMING_ENABLED = False
        global USE_CHECKPOINTS
        USE_CHECKPOINTS = False

    print("Building tree...")
    tree, found_words, tree_size, num_states_opened = construct_tree(
        guesses=[root_word_index],
        guess_results=[],
        table=table,
        depth=1,
        possible_answers=possible_answers,
    )

    print("Decision tree has been built")
    print(f"Tree size: {tree_size}")
    print(f"# states opened: {num_states_opened:,}")
    print("Found %d / %d words" % (len(found_words), len(answer_words)))
    if len(found_words) == len(answer_words):
        print("Success! Decision tree is full!")

    out_path = f"out/decision-trees/{dictionary}/{first_word}.json"
    with open(out_path, "w") as fp:
        json.dump(tree, fp, indent=4, sort_keys=True)
    print(f"Wrote tree to {out_path}")


def solve_all_cheating():
    words = read_all_answers()
    for word in tqdm(words):
        solve(dictionary="answers", first_word=word)


if __name__ == "__main__":
    coloredlogs.install()
    logging.basicConfig(level=logging.INFO)

    DEFAULT_ROOT_WORD = "serai"
    DEFAULT_CHEATING_ROOT_WORD = "crane"

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--first-word",
        type=str,
        default=None,
        help="The root word to try to build our tree",
    )
    parser.add_argument(
        "-d",
        "--dictionary",
        choices=["full", "answers", "asymmetric"],
        default="answers",
        help="The dictionary to use. Can either use the full dictionary (~13k words) or the cheating answers dictionary (~2300 words, default)",
    )
    parser.add_argument(
        "-m",
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth of the tree (number of guesses)"
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help="By default we look for any decision tree that solves in under max_depth. With this flag instead we are looking for the optimal decision tree."
    )
    args = parser.parse_args()

    first_word = DEFAULT_ROOT_WORD
    if args.first_word is None:
        if args.dictionary == "full" or args.dictionary == "asymmetric":
            first_word = DEFAULT_ROOT_WORD
        else:
            first_word = DEFAULT_CHEATING_ROOT_WORD
    else:
        first_word = args.first_word

    solve(dictionary=args.dictionary, first_word=first_word, max_depth=args.max_depth,
          find_optimal=args.find_optimal)
    # solve_all_cheating()
