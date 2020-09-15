import numpy as np
import pandas as pd
import networkx as nx
#from tqdm.noteboook import tqdm
from sklearn.metrics import r2_score
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
from joblib import Parallel, delayed

class Shap(object):
    
    def __init__(self , estimator,  train_or_test = 'test' , split = 0.25):
        self.estimator = estimator
        self.train_or_test = train_or_test
        self.split = split
        
    def fit(self , X , y):
        
        if self.train_or_test == 'test':
            X_train , X , y_train , y = train_test_split(X , y ,test_size= self.split)
            self.estimator.fit(X_train , y_train)
        elif self.train_or_test == 'train':
            self.estimator.fit(X , y)
        else:
            raise ValueError('The train_or_test parameter can only be "train" or "test"')

        explainer = shap.TreeExplainer(self.estimator , feature_perturbation = "tree_path_dependent")
        self.feature_importances_ = np.abs(explainer.shap_values(X)).mean(axis=0)        
        return
    
    def get_params(self):
        return self.estimator.get_params()
    
    def set_params(self , **params):
        self.estimator.set_params(**params)


def _fit_single(data , response , predictors , covariates , feat_selector):
    y = data[response]
    if response in predictors:
        X = data[predictors + covariates].drop(columns = response , inplace = False)
    else:
        X = data[predictors + covariates]

    feat_selector.fit(X = X.values , y = y.values)
    accepted = X.columns[feat_selector.support_]
    
    output = pd.Series(0 , index = X.columns , name = response)
    output[accepted] = 1
    
    return output

def _score(row , rf_regressor , data):
    if row.sum() > 0:
        X = data.loc[: , row == 1]
        y = data[row.name]
        rf = rf_regressor.fit(X, y)
        results = [r2_score(y , rf.predict(X)) , rf.oob_score_]
    else:
        results = [np.nan , np.nan]
    return pd.Series(results, index = ['R2_train' , 'R2_OOB'])

class EnsRegNet(object):
    """ Associate a set of "predictor" variables and "response" variables by fitting
    an ensemble regression method for each response ~ all the predictors.    
    
    Notes
    ----------
    
    The method used to select the most relevant predictors is called Boruta, please
    refer to the following link for a short explanation:
    https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
   
    Here are the link of the python packages which implement this method:
        - https://pypi.org/project/BorutaShap/
        - https://github.com/scikit-learn-contrib/boruta_py
    
        
    Parameters
    ----------
    responses : list of strings
        names of the response variable we want to include in the network
    
    predictors: list of strings
        names of potential predictors
        
    covariates: list of strings
        names of potential covariates. Default is []
        
    regressor: sklearn ensemble regression method
        for example sklearn.ensemble.RandomForestRegressor()
    
    verbose: int
        control the verbosity: the higher, the more messages. Default is 1.
    
    Attributes
    ----------
    selections_: DataFrame with binary values, shape (n_responses , n_predictors + n_covariates)
        selections_[response i , predictor j ]=1 indicates that the predictor j has been selected
        for the response i
            
    summary_: DataFrame
        gather several information for each regression (response vs predictors). For instance, the
        coefficient R2 for the training set...
    
    graph_: networkx object
    
    """
    
    def __init__(self , regressor , responses , predictors , covariates = [] , n_jobs = 1 , validation_regressor = None):
        self.responses = responses
        self.predictors = predictors
        self.covariates = covariates
        
        self.regressor = regressor
        self.validation_regressor = validation_regressor

        self.n_jobs = n_jobs
        
        # self.selections_ = pd.DataFrame(index = self.responses , columns = self.predictors + self.covariates)
        # self.summary_ = pd.DataFrame(index = self.responses , columns = ['R2 train' , 'R2 OOB'])
        # self.graph_ = None
    
    
    def fit(self , df , alpha = 0.01):
        """ Fit an ensemble regression model for each response using the dataframe df. Save the 
        selected features in the boolean dataframe self.selections_        

        Parameters
        ----------
        df : DataFrame
             columns must at least contain the predictors, the responses and the covariates
        
        alpha: float between 0 and 1.

        Returns
        -------
        None.

        """
        
        feat_selector = BorutaPy(self.regressor, n_estimators='auto', verbose=0 , alpha = alpha)
        
        parallel = Parallel(n_jobs=self.n_jobs) 
        selection = parallel(delayed(_fit_single)(df , 
                                                  resp , 
                                                  self.predictors ,
                                                  self.covariates , 
                                                  feat_selector)
                                 for resp in self.responses)
        
        self.selections_ = (pd.concat(selection , axis=1 ).T)[self.responses] 
        
        if not (self.validation_regressor is None):
            self.summary_ = self.selections_.apply( _score 
                                                   , args = (self.validation_regressor , df) 
                                                   , axis = 1)
        #for resp in tqdm(self.responses):
        # for resp in self.responses:  
        #     y = df[resp]
        #     if resp in self.predictors:
        #         X = df[self.predictors + self.covariates].drop(columns = resp , inplace = False)
        #     else:
        #         X = df[self.predictors + self.covariates]

        #     feat_selector.fit(X = X.values , y = y.values)
        #     accepted = X.columns[feat_selector.support_]
        #     self.selections_.loc[resp , X.columns] = 0
        #     self.selections_.loc[resp , accepted] = 1
  
        #     if len(accepted) > 0:
        #         rf = RandomForestRegressor(n_estimators=500 , max_depth=3, max_features =1/3, oob_score = True).fit(X[accepted] , y)
        #         self.summary_.loc[resp] = (r2_score(y , rf.predict(X[accepted])) , rf.oob_score_)   
        #     if self.verbose >= 1:
        #         print("Response : " , resp , "  Accepted predictors : " , list(accepted)
        #           , "  Undefined predictors : " , list(X.columns[feat_selector.support_weak_]))
        

        return
    
    def test(self , df_train , df_test):
        """
    
        Parameters
        ----------
        df_train : DataFrame
            columns must at least contain the predictors, the responses and the covariates
            
        df_test : DataFrame
            columns must at least contain the predictors, the responses and the covariates

        Returns
        -------
        None.

        """       
        self.summary_['R2_bis train'] , self.summary_['R2 test'] = np.nan , np.nan        
        #for resp in tqdm(self.responses):
        for resp in self.responses:
            temp = self.selections_.loc[resp]
            if temp.sum() > 0:
                X_train , y_train = df_train[self.selections_.columns[temp]] , df_train[resp]
                X_test  , y_test  = df_test[self.selections_.columns[temp]]  , df_test[resp]
                
                rf = RandomForestRegressor(max_depth=3, max_features =1/3, oob_score = True).fit(X_train , y_train)               
                self.summary_.loc[resp , ['R2_bis train' , 'R2 test']] = (r2_score(y_train , rf.predict(X_train)) , 
                                                                          r2_score(y_test  , rf.predict(X_test)))        
        return 

    # def get_adjacency(self):
    #     """ Compute the adjacency matrix from the dataframe self.selections_
        
    #     Returns
    #     -------
    #     numpy array, shape (n_responses , n_predictors)
    #         binary array, 1 indicates a link between the response and the predictor
    #         0 otherwise

    #     """
    #     return 1*(self.selections_.values[: , :len(self.predictors)])
    
    def create_graph(self , label_responses = None , label_predictors = None):
        self.graph_ = nx.Graph()
        for r in self.responses:
            for p in self.predictors:
                if self.selections_.loc[r , p] == 1:
                    self.graph_.add_node(r , label = label_responses)
                    self.graph_.add_node(p , label = label_predictors)
                    self.graph_.add_edge(r , p , weight = 1)
                    
        return nx.readwrite.json_graph.cytoscape_data(self.graph_)
