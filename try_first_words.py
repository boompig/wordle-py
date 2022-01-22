"""
Try the different possible first words
"""

from parse_data import read_past_answers, read_parsed_words
from play import eval_guess, UNSAFE_eval_guess, get_todays_answer
from typing import List, Dict
import random
from tqdm import tqdm
from collections import Counter


def is_word_matches(word: str, guess: str, guess_result: List[int]) -> bool:
    """
    :param guess: lowercase
    :param word: lowercase"""
    return UNSAFE_eval_guess(guess, word) == guess_result


def get_possible_words(guess: str, guess_result: List[int], words: List[str]) -> List[str]:
    return [word for word in words
            if is_word_matches(word, guess, guess_result)]


def get_num_possible_words(guess: str, guess_result: List[int], words: List[str]) -> int:
    """Return the possible words it could be after the guess"""
    count = 0
    guess = guess.lower()

    for word in words:
        if is_word_matches(word, guess, guess_result):
            count += 1
    return count


def eval_first_words(words: List[str], answer: str) -> Counter:
    """Evaluate all the possible first words for this answer.
    Return the number of possible words for each answer"""
    d = Counter()

    num_skipped = 0
    for word in tqdm(words):
        guess_result = eval_guess(guess=word, answer=answer)
        # if the guess is bad then we don't really need to get the number of possible words
        # we can approximate it to 10,000
        if guess_result == [0, 0, 0, 0, 0]:
            # print(f"skipping word {word}, no matches")
            num_skipped += 1
            n = 10_000
        else:
            n = get_num_possible_words(guess=word, guess_result=guess_result, words=words)
            if n > 10_000:
                # to keep things consistent, the maximum number of possible words is limited to 10_000
                n = 10_000
        d[word] = n
    print(f"Skipped exact counts for {num_skipped} words which had a bad match")
    return d


if __name__ == "__main__":

    test_possible_words = False
    if test_possible_words:
        answer = get_todays_answer()
        print(f"answer: {answer}")

        words = read_parsed_words()
        guess = "audio"
        print(f"guess: {guess}")

        guess_result = eval_guess(guess, answer)
        n = get_num_possible_words(guess, guess_result, words)
        print(f"# possible words: {n}")

    test_first_word_one_answer = False
    if test_first_word_one_answer:
        answer = get_todays_answer()
        print(f"answer: {answer}")
        words = read_parsed_words()

        print(f"Testing all {len(words)} words as first guess...")
        d = eval_first_words(words, answer)

        print("Here are the top most efficient words:")
        best_candidates = d.most_common()
        best_candidates.reverse()
        i = 0
        for word, num_remaining in best_candidates[30:]:
            guess_result = eval_guess(word, answer)
            cand_words = get_possible_words(word, guess_result, words)
            print(f"{i + 1}. {word} -> {num_remaining} possible words ; eval = {guess_result} ; candidate words = {', '.join(cand_words[:4])}...")
            if i > 30:
                break
            i += 1

    test_first_word_all_answers = True
    if test_first_word_all_answers:
        answers = read_past_answers()
        for i, answer in enumerate(answers):
            print(f"Day #{i} - answer: {answer}")
            words = read_parsed_words()

            print(f"Testing all {len(words)} words as first guess...")
            d = eval_first_words(words, answer)

            print("Here are the top most efficient words:")
            best_candidates = d.most_common()
            best_candidates.reverse()
            i = 0
            for word, num_remaining in best_candidates[30:]:
                guess_result = eval_guess(word, answer)
                cand_words = get_possible_words(word, guess_result, words)
                print(f"{i + 1}. {word} -> {num_remaining} possible words ; eval = {guess_result} ; candidate words = {', '.join(cand_words[:4])}...")
                if i > 30:
                    break
                i += 1

