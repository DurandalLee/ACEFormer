# ACEFormer

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
[![ACEFormer](https://img.shields.io/badge/ACEFormer-brightgreen.svg?style=plastic)](https://github.com/DurandalLee/ACEFormer)

This is the origin Pytorch implementation of ACEFormer in the following paper:
[An End-to-End Structure with Novel Position Mechanism and Improved EMD for Stock Forecasting](https://arxiv.org/abs/2404.07969)

## Table of Contents
- [ACEFormer](#aceformer)
	- [Requirements](#requirements)
	- [Data](#data)
	- [Usage](#usage)

## Requirements

- Python 3.7
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

## Data

The stock dataset used in the paper can be downloaded in the repo [Stock Data](https://github.com/DurandalLee/ACEFormer/tree/main/stockdata).

Two real-world datasets, which are NASDAQ100 and SPY500, from US stock markets spanning over ten years.
The NASDAQ100 is a stock market index made up of 102 equity stocks of non-financial companies from the NASDAQ.
The SPY500 is Standard and Poor's 500, which is a stock market index tracking the stock performance of 500 large companies listed on stock exchanges in the United States.
The [historical data](https://www.investing.com/) ranging from Jan-03-2012 to Jan-28-2022 for our experiments.

## Usage
The training script for *ACEFormer* is the [ACEFormer.py](https://github.com/DurandalLee/ACEFormer/tree/main/ACEFormer.py). A sample command to execute the script is as follows:

```bash
# NDX100 
python cpu 5 ./result ./stockdata/NDX100.csv 1 2000
```

Additionally, you can refer to the example in the [example.ipynb](https://github.com/DurandalLee/ACEFormer/tree/main/example.ipynb) to understand the model training process.