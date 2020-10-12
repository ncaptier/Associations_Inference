# Associations_Inference

This repository proposes a new method to infer relevant associations within a set of features. It infers relevant predictors for several regression problems combining the Boruta method [1] and the Shapley values for feature importance [2].   

This repository also proposes a python implementation for the TIGRESS method which solves the same kind of problems.

## Experiments

In order to illustrate our method, we provide a jupyter notebook which applies it to the DREAM4 In silico size 100 multifactorial subchallenge. We compare the results with state-of-the-art methods such as TIGRESS [3] and GENIE3 [4] **(ongoing building)**.

## Data 

The training data sets and the gold standard data sets for the DREAM4 challenge can be downloaded [here](http://dreamchallenges.org/project/dream4-in-silico-network-challenge/).

## Requirements

To run this algorithm as well as the jupyter notebook, one will need the following python packages:
* [boruta](https://github.com/scikit-learn-contrib/boruta_py)
* joblib
* matplotlib.pyplot
* networkx
* numpy
* pandas
* scikit-learn
* [shap](https://github.com/slundberg/shap)

## Acknowledgements

This package was created as a part of Master stage by Nicolas Captier in the [Computational Systems Biology of Cancer group](http://sysbio.curie.fr) of Institut Curie.

## References
[1] "Feature selection with boruta package" - Kursa and Rudnicki 2010   
[2] "A unified approach to interpreting model predictions" - Lundberg et al. 2017   
[3] "TIGRESS: Trustful Inference of Gene REgulation using Stability Selection" - Haury et al. 2012   
[4] "Inferring regulatory networks from expression data using tree-based methods" - Vân Anh Huynh-Thu et al. 2010    