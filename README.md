# ACEFormer

# ACEFormer

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
[![ACEFormer](https://img.shields.io/badge/ACEFormer-brightgreen.svg?style=plastic)](https://github.com/DurandalLee/ACEFormer)

This is the origin Pytorch implementation of ACEFormer in the following paper:
An End-To-End Structure with Improved EMD and Temporal Perception Mechanism for Stock Forecasting

## Table of Contents

- [ACEFormer](#aceformer)
	- [Table of Contents](#table-of-contents)
	- [Background](#background)
	- [ACEEMD](#aceemd)
	- [Requirements](#requirements)
	- [Data](#data)
	- [Usage](#usage)

## Background

The highly volatile and non-linear nature of stock data often leads to rapid oscillations that can be considered as noise and require removal. 
To address this issue, we introduce a noise reduction algorithm called aliased complete ensemble empirical mode decomposition with adaptive noise (ACEEMD). 
Also we present the ACEFormer, which includes ACEEMD, the temporal perception mechanism, and the attention mechanisms. The temporal perception mechanism can overcome the weak extraction ability of positional information in Informer.

<p align="center">
<img src=".\image\ACEFormer.png"/>
<br><br>
<b>Figure 1.</b> The architecture of ACEFormer.
</p>

## ACEEMD

The ACEEMD can improve the fitting of the original curve by mitigating the endpoint effect and preserving outliers in stock data, which can have significant impacts on trading.

<p align="center">
<img src=".\image\all_ACEEMD.png"/>
<br><br>
<b>Figure 2.</b> The ACEEMD.
</p>

$x(t)$ refers to the input data, i.e. the stock data.
$n^{[i]}$ represents the $i$-th Gaussian noise, where the total number of noises $m$ is an adjustable parameter and the default value is 5.
$E(\cdot)$ denotes the first-order IMF component of the signal in parentheses.
$pe_i$ and $pm_i$ both represent the result of the input data and a Gaussian noise, but the difference between them is that the Gaussian noise they add is opposite in sign to each other.
The generating function $AM(pe_i,pm_i)$ is used to denoise the data and is also the core of ACEEMD.
$IMF_1$ represents the first-order IMF component of ACEEMD, which is the eliminable noise component in the input data.
$r^i_1(t)$ represents the denoised data of the input data with the $i$-th group of added Gaussian noise, and $r_1(t)$ represents the denoised data obtained by processing the input data with ACEEMD.

## Requirements

- Python 3.7
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

## Data

The stock dataset used in the paper can be downloaded in the repo [Stock Data](https://github.com/DurandalLee/ACEFormer/tree/main/data).

Two real-world datasets, which are NASDAQ100 and SPY500, from US stock markets spanning over ten years.
The NASDAQ100 is a stock market index made up of 102 equity stocks of non-financial companies from the NASDAQ.
The SPY500 is Standard and Poor's 500, which is a stock market index tracking the stock performance of 500 large companies listed on stock exchanges in the United States.
The [historical data](https://www.investing.com/) ranging from Jan-03-2012 to Jan-28-2022 for our experiments.

## Usage

Commands for training and testing the model with *ACEFormer* on Dataset NDX100.csv and SPY500.csv respectively:

```bash
# NDX100 
python ACEFormer.py cuda:0 5 ./result ./data/NDX100.csv 1 2000

# SPY500.csv
python ACEFormer.py cuda:0 5 ./result ./data/SPY100.csv 1 2000
```
