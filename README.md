# Freq bias benchmark

A proposed benchmark to evaluate langauge model's generation of long-tail knowledge


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

### gather descriptive freq in train and genration set
```bash
python compare_freq.py
```

### oov analysis


