# Intrusion Detection System
For the course Key Topics in Artificial Intelligence, we explored several machine learning models, i.e., Deep Neural Networks (DNNs) and Random Forests (RFs). In this repository, we present the python code. 

The dataset that was used, UNSW-NB15, can be downloaded [here](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15). It is required to place the files `/UNSW_NB15_testing-set.csv` and `/UNSW_NB15_training-set.csv` in the `/dataset` folder. 

For the DNN models, we used the following feature selection strategies
| **Strategy** | **Description**  | **# features**  |
|---|---|---|   
| Default | input features were one-hot encoded  | 196 |
| FAMD | Factorial Analysis of Mixed Data | 152 |
| PCA | Principal Component Analysis | 18 |
| K-means | Feature ranking based on K-means homogeneity scores | 24 |
| Top n | Reducing categorical features based on value counts, clamping, and one-hot encoding | 56 |

epochs = 100
batch_size = 1000

## Binary DNN model
```
input_shape -> 20 -> 20 -> 20 -> 20 -> 1
```

## Multi-class DNN model

```
input_shape -> 100 -> 100 -> 100 -> 100 -> 10
```
