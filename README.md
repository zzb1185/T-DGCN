This is a TensorFlow implementation of A Deep Learning Model of Coalbed Methane Production Prediction Considering Time, Space, and Geological Features.

# The manuscript
#### A Deep Learning Model of Coalbed Methane Production Prediction Considering Time, Space, and Geological Features


# The Code
## Requirements:
* tensorflow(The best is tensorflow GPU = = 1.14.0)
* scipy
* numpy
* matplotlib
* pandas
* math

## Geological features
*  DTW_matrix.py

        You just need to run it
        Generate an adjacency matrix with spatial and geological features.
        The generated file is saved in "data \ testdata \ test_ohe. CSV"
* Spatial_matrix.py

        Generate an adjacency matrix with spatial features.
        The generated file is saved in "data \ testdata \ test_oh1. CSV"
## T-DGCN model
* Python main.py

        You just need to run it.
        And provide the following parameter adjustment interface:
        * learning_rate
        * training_epoch
        * gru_units
        * seq_len
        * pre_len
        * train_rate
        * batch_size
        * dataset
        * model_name



## Implement

In the CBM dataset, we set the parameters seq_len to 7 days and pre_len to 1, 3, 5, 7 days. 

## Data Description
### As the CBM data is classified and controlled by the state, the original data used in the paper is not included in this project.


    However, we prepared some test data to verify the usability of the code. 
    Due to the small amount of data, the performance of the model did not meet the expectations. 
    Please refer to the manuscript for the theoretical accuracy of the model. 
    In addition, for the sake of confidentiality, we have moderately transformed the data, so these data seem to be slightly different from common sense.


# Special Thanks
Special thanks to Zhao et al. For their research papers:<br>
T-GCN: A temporal graph convolutional network for traffic prediction
The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901   or  https://arxiv.org/abs/1811.05320 

# e-mail：1185702573@qq.com

✨
