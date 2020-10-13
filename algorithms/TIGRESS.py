import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.random import sample_without_replacement
from joblib import Parallel, delayed
from sklearn.linear_model import lasso_path , lars_path , enet_path 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

""" This code proposes a python implementation of the TIGRESS method developped by Haury et al. in 2012 for inferring
gene regulatory networks (https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-145)."""

def _bootstrap_generator(R, bootstrap , X , y):
    """ Yields R bootstrap samples from X and y
    
    Parameters
    ----------
    R : int
        number of boostrap samples.
        
    bootstrap : float <= 1
        size of each bootstrap sample (as a fraction of the size of X).
        
    X : array, shape (n_samples , n_features)
        
    y : array, shape (n_samples)

    Yields
    ------
    array, shape (floor(boostrap*n_samples) ,n_features)

    array, shape (floor(boostrap*n_samples))
    
    """
    n_samples = X.shape[0]
    n_subsamples = np.floor(bootstrap*n_samples).astype(int)
    for _ in range(R):
        subsample = sample_without_replacement(n_samples, n_subsamples)
        yield (X[subsample , :] , y[subsample] )

def _subsampling_generator(R , alpha , X , y):
    """ Yields R subsamples from X and y. Randomly splits the data sets into to halves of equal 
    (or approximately equal) size and multiplies each row by a random number uniformly sampled on the interval [alpha , 1].
    
    Parameters
    ----------
    R : int
        number of subsampling operations.
        
    alpha : float < 1
        parameter for the uniform probability law     
        
    X : array, shape (n_samples , n_features)
        
    y : array, shape (n_samples)

    Yields
    ------
    array,  shape (0.5*n_samples ,n_features)

    array, shape (0.5*n_samples)

    """
    for _ in range(int(R/2)):
        X1 , X2 ,  y1 , y2 = train_test_split(X , y , test_size = 0.5)
        yield (np.random.uniform(alpha , 1 , size = (X1.shape[0] , 1))*X1 , y1)
        yield (np.random.uniform(alpha , 1 , size = (X2.shape[0] , 1))*X2 , y2)

def _fit_bootstrap_sample(X , y, func , L):
    """ Computes the regularization path for the regression y ~ X.
    
    Parameters
    ----------
    X : array, shape (n_samples , n_features)

    y : array, shape (n_samples)

    func : string
         the function used for computing the regularization path
         (either 'lasso', 'elasticnet', or 'lars').
        
    L : int
        length of the path.

    Returns
    -------
    array, shape (n_features , L) 
        0 if the coefficient is null and 1 otherwise.

    """
    if func == 'lasso':
         _, coef_path, _ = lasso_path(X, y , n_alphas = L)
    elif func == 'elasticnet':
          _, coef_path, _ = enet_path(X, y , nalphas = L)
    elif func == 'lars':
         _, _, coef_path = lars_path(X, y , max_iter = L-1)

    return 1*(coef_path != 0)


class StabilizedSelection(object):    
    """ Performs stability selection using the regularization pathes 
    (see. https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-145).
    
    Parameters
    -----------
    
    func : string
        the function used for computing the regularization path
         (either 'lasso', 'elasticnet', or 'lars').
         
    scoring : string
        if "area" the area under the frequency curve is computed and if "original" the maximum frequency is used.
        
    R : int
        number of resampling operations.
    
    L : int
        length of the regularization path.
    
    resampling : string
        either "bootstrap" strategy or "subsamples".
        
    alpha : float < 1
        see _subsampling_generator function.
        
    bootstrap : float <= 1
        see _bootstrap_generator function.
        
    parallel : optional
        Default is None.
        
    n_jobs : int, optional
        number of jobs to run in parallel. -1 means using all processors.
        See the joblib package documentation for more explanations. Default is 1.
    
    verbose: int, optional
        control the verbosity: the higher, the more messages. Default is 2.     
        
    """
    
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
        """       
        1. Computes the regularization pathes for each resampling (R times in total)
        2. For each pair (feature , lambda), computes the frequency with which the coefficient associated to
           the feature and to the regularization coefficient "lambda" was non zero.
        3. Scores each feature either with the "area under the frequency curve" or with ...
    
        Parameters
        ----------
        X : array, shape (n_samples , n_features)
            
        y : array, shape (n_samples)

        Returns
        -------
        None.

        """
        if self.resampling == 'bootstrap':
            resampling_samples = _bootstrap_generator(self.R, self.bootstrap , X , y)
        elif self.resampling == 'subsamples':
            resampling_samples = _subsampling_generator(self.R , self.alpha, X, y)
        
        # 1. Computes the regularization pathes
        if self.parallel is None:
            self.parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        selection = self.parallel(delayed(_fit_bootstrap_sample)(X = subsample[0] ,
                                                                 y = subsample[1] , 
                                                                 func = self.func ,
                                                                 L = self.L)
                             for subsample in resampling_samples)
        
        # 2. Computes the frequencies over the R resamplings
        self.pathes_ = (1/len(selection))*sum(selection)
        
        # 3. Computes a score for each feature using the stabilized frequencies along the regularization path.
        if self.scoring == 'area':
            self.scores_ = np.mean(self.pathes_ , axis = 1)
        elif self.scoring == 'original':
            self.scores_ = np.amax(self.pathes_ , axis = 1)
            
        return
    
    def plot_pathes(self , ax = None):
        """ Plots stability pathes, i.e for each feature plots the frequency of selection
        with respect to the path step.
        
        Parameters
        ----------
        ax :  matplotlib.axes, optional
            Default is None.

        Returns
        -------
        None.

        """
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
    """ Associates a set of "predictor" variables and "response" variables by combining
    regularized linear regression and stability selection.
    
    Note
    ------    
    For details, please refer to the article :
        https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-6-145
        
    Parameters
    ---------
    
    responses : list of strings
        names of the response variables
    
    predictors : list of strings
        names of the potential predictors
    
    covariates : list of strings, optional
        names of the potential covariates. Default is []
    
    n_jobs : int, optional
        number of jobs to run in parallel. -1 means using all processors.
        See the joblib package documentation for more explanations. Default is -1.
    
    verbose: int, optional
        control the verbosity: the higher, the more messages. Default is 0.
    
    R : int, optional
        Default is 4000
    
    L : int, optional
        length of the regularization path. Default is 4.
        
    func : string, optional
        function for computing the regularization path. Default is 'lars'.
    
    scoring : string, optional.
        scoring method. Default is "area".
    
    alpha : float <= 1, optional
        see _subsampling_generator function. Default is 0.4.
    
    resampling : string, optional
        define the resampling strategy. Default is 'subsamples'.
    
    bootstrap : float <= 1
        see _bootstrap_generator function.
    
    Attributes
    ---------
    
    scores_ : DataFrame, shape (n_responses , n_predictors)
    
    Note
    -------
    The default values of the parameters are inspired by the original article descirbing TIGRESS method.
    
    """
    
    def __init__(self , responses , predictors , covariates = [] , n_jobs = -1 , verbose = 0
                 , R = 4000 , L = 4 , func = 'lars', scoring = 'area' , alpha = 0.4 
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
        
        self.scores_ = None
        return
    
    def fit(self , df , normalize = False):
        """ Fits a Stabilizedselection method for each response using the dataframe df. Saves the stability
        scores associated with the potential predictors in the dataframe self.scores_.
        
        Parameters
        ----------
        df : DataFrame, shape (n_samples, n_responses + n_predictors + n_covariates)
            
        normalize : boolean, optional
            Default is False.

        Returns
        -------
        None.

        """
        
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
                                                 , R = self.R , L = self.L , alpha = self.alpha , resampling = self.resampling 
                                                 , bootstrap = self.bootstrap)
                    stable.fit(X.values , y.values)
                    temp.append(pd.Series(stable.scores_ , index = X.columns , name = resp))
        self.scores_ = (pd.concat(temp , axis = 1).T)[self.predictors] 
        return 
    
    