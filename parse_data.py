"""
This file reads the raw data and parses it for faster and easier future reading
"""

from typing import List, Optional
from datetime import datetime
import pandas as pd
import pickle
import os.path


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_PARSED_ANSWERS_FILE = os.path.join(
    BASE_DIR, "data-parsed/wordle-answers.parquet"
)
DEFAULT_PARSED_WORDS_FILE = os.path.join(BASE_DIR, "data-parsed/wordle-words.pickle")


def read_wordle_answers_raw(fname: str) -> pd.DataFrame:
    """
    Read the wordle answers in the format that I found them.
    Return a dataframe of all the information.
    """

    # line format: (Month) (MonthDay) (Year) ("Day") (DayNum) (Word)
    # Month: short name (3 letters)
    # Year: 4 digits
    # MonthDay: 0-padded 2-digit number
    # DayNum: 0-padded 3-digit number

    answers = []  # type: List[dict]

    with open(fname) as fp:
        for line in fp:
            if line[-1] == "\n":
                line = line[:-1]
            if line == "" or line[0] == "#":
                continue
            month, day_of_month, year, _, day_num, answer = line.split(" ")
            # we only convert to integer here to sanity-check
            year = int(year)
            day_of_month = int(day_of_month)
            assert len(month) == 3
            assert len(answer) == 5, answer
            answer_date = datetime.strptime(
                f"{year} {month} {day_of_month}", "%Y %b %d"
            )
            day_num = int(day_num)
            row = {"date": answer_date, "day": day_num, "answer": answer}
            answers.append(row)

    return pd.DataFrame(answers)


def read_past_answers(fname: Optional[str] = None) -> List[str]:
    """
    Read all the Wordle answers on or before today's date
    """
    all_answers = read_parsed_answers(fname)
    now = datetime.now()
    todays_date = datetime(now.year, now.month, now.day)
    todays_date_pretty = datetime.strftime(todays_date, "%Y %b %d")
    print(f"Reading all answers on or before {todays_date_pretty}...")
    past_answers = all_answers[all_answers["date"] <= todays_date]
    return past_answers["answer"].tolist()


def read_all_answers(fname: Optional[str] = None) -> List[str]:
    all_answers = read_parsed_answers(fname)
    return [w.lower() for w in all_answers["answer"].tolist()]


def read_wordle_words_raw(fname: str) -> List[str]:
    words = []
    with open(fname) as fp:
        for line in fp:
            if line[-1] == "\n":
                line = line[:-1]
            if line == "" or line[0] == "#":
                continue
            # should be a 5-letter word
            assert len(line) == 5
            words.append(line)
    return words


def read_parsed_answers(fname: Optional[str] = None) -> pd.DataFrame:
    """Return the parsed answers as a dataframe.
    That dataframe will have the following fields:
        - date -> datetime object representing the date of the puzzle
        - day -> a numeric day (0-indexed)
        - answer -> all-caps 5-letter word that is the answer on that day
    :param fname: Should be a parquet file
    """
    if fname is None:
        fname = DEFAULT_PARSED_ANSWERS_FILE
    return pd.read_parquet(fname)


def read_parsed_words(fname: Optional[str] = None) -> List[str]:
    """
    Read the parsed words file
    :param fname: Should be a pickle file
    """
    if fname is None:
        fname = DEFAULT_PARSED_WORDS_FILE
    with open(fname, "rb") as fp:
        return pickle.load(fp)


if __name__ == "__main__":
    # parses answers
    read_write_answers = True
    if read_write_answers:
        print("Reading Wordle answers...")
        answers = read_wordle_answers_raw("data-raw/wordle-answers-future.txt")
        print(f"Read {len(answers)} answers")
        answers.to_parquet(DEFAULT_PARSED_ANSWERS_FILE)
        print("saved to file")

    # parses the dictionary file
    read_write_words = False
    if read_write_words:
        print("Reading the wordle dictionary...")
        words = read_wordle_words_raw("data-raw/wordle-words.txt")
        print(f"Read {len(words)} words")
        # save these in pickle format for faster loading
        with open(DEFAULT_PARSED_WORDS_FILE, "wb") as fp:
            pickle.dump(words, fp)
        print("saved to file")
