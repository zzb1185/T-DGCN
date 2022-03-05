This is a TensorFlow implementation of A Deep Learning Model of Coalbed Methane Production Prediction Considering Time, Space, and Geological Features.

# The manuscript
### A Deep Learning Model of Coalbed Methane Production Prediction Considering Time, Space, and Geological Features


# The Code
## Requirements:
* tensorflow(The best is tensorflow GPU = = 1.14.0)
* scipy
* numpy
* matplotlib
* pandas
* math

## Geological features
* 01 Spatial nearest neighbor relation matrix.ipynb
* 02 Normalization.py
* 03 DTW.py
* 04 Combination.py

## T-DGCN model
Python main.py

Our baselines included: <br>

(1) Autoregressive Integrated Moving Average model (ARIMA)<br>
(2) Support Vector Regression model (SVR)<br>
(3) Graph Convolutional Network model (GCN)<br>

The python implementations of HA/ARIMA/SVR models were in the baselines.py; The GCN was in gcn.py respective.

The T-GCN model was in the tgcn.py


## Implement

In the CBM dataset, we set the parameters seq_len to 7 days and pre_len to 1, 3, 5, 7 days. 

## Data Description
### As the CBM data is classified and controlled by the state, the original data used in the paper is not included in this project.
### SORRY
However, it contains two sets of similar data sets. The author's idea is affected by the research in the field of transportation. Therefore, it includes the following data sets:<br>
(1) SZ-taxi. This dataset was the taxi trajectory of Shenzhen from Jan. 1 to Jan. 31, 2015. We selected 156 major roads of Luohu District as the study area.<br>
(2) Los-loop. This dataset was collected in the highway of Los Angeles County in real time by loop detectors. We selected 207 sensors and its traffic speed from Mar.1 to Mar.7, 2012

In order to use the model, we need
* A N by N adjacency matrix, which describes the spatial relationship between roads, 
* A N by D feature matrix, which describes the speed change over time on the roads.

# Special Thanks
Special thanks to Zhao et al. For their research papers:<br>
T-GCN: A temporal graph convolutional network for traffic prediction
The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901   or  https://arxiv.org/abs/1811.05320 

# Moreover
Miss You Baby! :heartpulse:

# e-mail：zzb1185@126.com

