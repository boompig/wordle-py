# Running

This project contains a variety of scripts that you can run:

- `parse_data.py` -> parse the raw data contained in `data-raw` and put it into `data-parsed`
- `possibilities_table.py` -> compute the possibilities matrix
- `play.py` -> play Wordle on the command line with today's word
- `decision_tree.py` -> assists you in solving

## Solver

Run with `-h` to see the various options

## Installing Dependencies

Run `poetry install` to install dependencies

## Linting

Run `poetry run black *.py`
Then run `poetry run mypy *.py --ignore-missing-imports`

## Notebooks

There are several Jupyter notebooks in the `notebooks` directory.
You can run them with `jupyter lab`.
