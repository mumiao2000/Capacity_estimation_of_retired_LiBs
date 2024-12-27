## Introduction

This repo is the official implementation for the paper: Capacity estimation of retired lithium-ion batteries using random charging segments from massive real-world data (CRPS).

In this study, we compiled the two largest known datasets of retired lithium-ion batteries and developed a tailored neural network model capable of directly capturing both long-term and short-term temporal patterns. By utilizing randomly segmented charging curves as input, the proposed model reduces the cost of charging-discharging tests in practical applications.

**NOTE!**: If you want to use our datasets, please name them as "**LINKE Dataset**" ([Lab for Intelligent Networking and Knowledge Engineering, USTC](https://linke.ustc.edu.cn)).

<p align="center">
<img src=".\Fig\Pipeline.jpg" width = "800" height = "" alt="" align=center />
</p>

## Overall Architecture

The substantial variations among battery cells present two critical challenges: complex charging-discharging profiles, and imbalanced capacity distribution. This study introduced a sampling method to address the issue of imbalanced capacity distribution. Additionally, 2D-CNN was designed to capture both long-term and short-term patterns in time series data, facilitating the effective feature extraction from complex partial charging curves.

<p align="center">
<img src=".\Fig\2D-CNN.jpg" width = "800" height = "" alt="" align=center />
</p>

## Usage 

1. Install Pytorch and other necessary dependencies.

``` bash
pip install pandas numpy tqdm pytorch
```

2. Unzip ```raw_data.zip```:

``` bash
unzip AOC15/data/raw_data.zip -d AOC15/data/
unzip AOC23/data/raw_data.zip -d AOC23/data/
```

3. Change directory to AOC23 or AOC15:

``` bash
cd AOC23 # or cd AOC15
```

4. Generate dataset:

``` bash
python 1_generate_datast.py
```

5. Train the model:

``` bash
python 2_training.py
```

6. Customize your own model:

Create a new class in ```model.py``` to implement your model, and create an instance of it on **Line 54** in ```2_training.py```.

``` python
net = model.Custom_Model(seq_len).to(device)
```

## Accuracy of different methods

We evaluate the estimation accuracy of 2D-CNN against other five methods on AOC23 \& AOC15. The comparative methods are MLP, LSTM, TCN, 1D-CNN and GAF-CNN. Experimental results show that the performance of our proposed 2D-CNN is the best among these methods.

### Accuracy on AOC23

<p align="center">
<img src=".\Fig\23_Result.jpg" width = "800" height = "" alt="" align=center />
</p>

### Accuracy on AOC15

<p align="center">
<img src=".\Fig\15_Result.jpg" width = "800" height = "" alt="" align=center />
</p>

## Performance under different segment length

To further explore the effect of random segment length on capacity estimation, datasets with \(r\) values ranging from 0.4 to 1.0 were generated. Among all methods, the proposed 2D-CNN achieves the lowest RMSE, MAE, and MAPE in the vast majority of cases.

<p align="center">
<img src=".\Fig\Result.jpg" width = "800" height = "" alt="" align=center />
</p>

## Citation

If you find this repo helpful, please cite our paper.



## Concat

If you have any questions or want to use the code, please contact pengfeizhou@mail.ustc.edu.cn
