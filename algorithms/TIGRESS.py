import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.random import sample_without_replacement
from joblib import Parallel, delayed
from sklearn.linear_model import lasso_path , lars_path , enet_path 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _bootstrap_generator(R, bootstrap , X , y):
    n_samples = X.shape[0]
    n_subsamples = np.floor(bootstrap*n_samples).astype(int)
    for _ in range(R):
        subsample = sample_without_replacement(n_samples, n_subsamples)
        yield (X[subsample , :] , y[subsample] )

def _subsampling_generator(R , alpha , X , y):
    for _ in range(int(R/2)):
        X1 , X2 ,  y1 , y2 = train_test_split(X , y , test_size = 0.5)
        yield (np.random.uniform(alpha , 1 , size = (X1.shape[0] , 1))*X1 , y1)
        yield (np.random.uniform(alpha , 1 , size = (X2.shape[0] , 1))*X2 , y2)

def _fit_bootstrap_sample(X , y, func , L):
    if func == 'lasso':
         _, coef_path, _ = lasso_path(X, y)
    elif func == 'elasticnet':
          _, coef_path, _ = enet_path(X, y)
    elif func == 'lars':
         _, _, coef_path = lars_path(X, y , max_iter = L)   

    return 1*(coef_path != 0)


class StabilizedSelection(object):
    
    def __init__(self , func , scoring , R , L , resampling , alpha , bootstrap 
                 , parallel = None , n_jobs = 1 , verbose = 2):       
        self.func = func
        self.scoring = scoring
        self.R = R
        self.L = L
        self.resampling = resampling
        self.alpha = alpha
        self.bootstrap = bootstrap
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose
        return
    
    def fit(self, X , y):
        
        if self.resampling == 'bootstrap':
            resampling_samples = _bootstrap_generator(self.R, self.bootstrap , X , y)
        elif self.resampling == 'subsamples':
            resampling_samples = _subsampling_generator(self.R , self.alpha, X, y)
        
        if self.parallel is None:
            self.parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        selection = self.parallel(delayed(_fit_bootstrap_sample)(X = subsample[0] ,
                                                                 y = subsample[1] , 
                                                                 func = self.func,
                                                                 L = self.L)
                             for subsample in resampling_samples)

        self.pathes_ = (1/len(selection))*sum(selection)
        
        if self.scoring == 'area':
            self.scores_ = np.mean(self.pathes_ , axis = 1)
        elif self.scoring == 'original':
            self.scores_ = np.amax(self.pathes_ , axis = 1)
        
        return
    
    def plot_pathes(self , ax = None):
        if ax is None:
            plt.figure(figsize = (10 , 7))
            plt.plot(self.pathes_.T)
            plt.title("Stability pathes - " + self.func , fontsize = 14)
            plt.xlabel("Path steps" , fontsize = 12)
            plt.ylabel("frequency of selection" , fontsize = 12)
        else:
            ax.plot(self.pathes_.T)
            ax.set_title("Stability pathes - " + self.func)
            ax.set_xlabel("Path steps")
            ax.set_ylabel("frequency of selection")
        return 

class TIGRESS(object):
    
    def __init__(self , responses , predictors , covariates = [] , n_jobs = -1 , verbose = 0
                 , R = 1000 , L = 3 , func = 'lars', scoring = 'area' , alpha = 0.4 
                 , resampling = 'subsamples' , bootstrap = 1):
        self.responses = responses
        self.predictors = predictors
        self.covariates = covariates
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.R = R
        self.L = L
        self.func = func
        self.scoring = scoring
        self.alpha = alpha
        self.resampling = resampling
        self.bootstrap = bootstrap
        return
    
    def fit(self , df , normalize = False):
        
        if normalize:
            df = pd.DataFrame(StandardScaler().fit_transform(df) , index = df.index , columns = df.columns)
            
        with Parallel(n_jobs=self.n_jobs , verbose = self.verbose) as parallel:
            temp = []
            for resp in self.responses:
                    y = df[resp]
                    if resp in self.predictors:
                        X = df[self.predictors + self.covariates].drop(columns = resp , inplace = False)
                    else:
                        X = df[self.predictors + self.covariates]
                    
                    stable = StabilizedSelection(parallel = parallel , func = self.func , scoring = self.scoring
                                                 , R = self.R , alpha = self.alpha , resampling = self.resampling 
                                                 , bootstrap = self.bootstrap , L = self.L)
                    stable.fit(X.values , y.values)
                    temp.append(pd.Series(stable.scores_ , index = X.columns , name = resp))
        self.scores_ = (pd.concat(temp , axis=1 ).T)[self.responses] 
        return 
    
    