# ML-intro-2022-9
[![Python package](https://github.com/Feosen/ml-intro-2022-9/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/Feosen/ml-intro-2022-9/actions/workflows/python-package.yml)

## Usage
This package allows you to train model for house price prediction.
1. Clone this repository to your machine.
2. Download [Melbourne housing dataset](https://www.kaggle.com/anthonypino/melbourne-housing-market) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Activate shell:
```sh
poetry shell
```
6. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
8. Deactivate shell:
```sh
exit
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [tox](https://github.com/tox-dev/tox): 
```
tox
```
