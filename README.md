# Freq bias benchmark

A proposed benchmark to evaluate langauge model's generation of long-tail knowledge
You could add many tools to this template (use [Typer](https://typer.tiangolo.com/)
for your CLI, [pre-commit](https://pre-commit.com/) hooks, [pylint](https://pylint.pycqa.org), etc.).

## Installation

see `requirements.txt` 

```bash
conda create -n myenv
conda activate myenv
conda install python pip
pip install .
```


## Data preparation 

segment utterances in train data for generation

```bash
python prepare_generation.py
```

## Generation
### autoregressive model
LSTM: next token prediction, adapted from [Lexical-benchmark](https://github.com/Jing-L97/Lexical-benchmark)

## Results analysis
### oov analysis


### freq proportion analysis