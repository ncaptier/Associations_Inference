import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import r2_score
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
import shap
from joblib import Parallel, delayed

class Shap(object):
    """ Adds Shap feature importance to classical tree-based machine learning models.
    
    This is a technical class which will allow us to later combine Shap and Boruta method.
    
    Parameters
    ----------
    estimator : model object
        The tree-based machine learning model that we want to explain (ex : XGBoost or almost
        all the tree-based models from scikit-learn).
        
    train_or_test : string, optional
        if 'test' operate a train_test split during fitting and compute shap features importance
        on test set. Default is 'test'.
        
    split : 0 < float < 1, optional
        parameter for the train_test split operation if train_or_test = 'test'.
        Default is 0.25.
    
    Attributes
    ----------
    
    feature_importances_ : array, shape (n_features)
        mean absolute Shapley values.
        
    """
    
    def __init__(self , estimator,  train_or_test = 'test' , split = 0.25):
        self.estimator = estimator
        self.train_or_test = train_or_test
        self.split = split
        
        self.feature_importances_ = None 
        
    def fit(self , X , y):
        """ Fits the estimator model with (X , y) and compute the associated mean absolute
        Shapley values.
        
        If self.train_or_test = "test" first operate a train_test split on (X , y) and then
        compute the feature importances.
        
        Parameters
        ----------
        X : array-like of shape (n_samples , n_features)
            training samples
            
        y : array-like of shape (n_samples)
            target values

        Returns
        -------
        None.

        """
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

########################################################################################

def _fit_single(data , response , predictors , covariates , feat_selector):
    """ Uses a feature selector tool to select all the relevant features for the regression problem
    response ~ predictors + covariates 
    
    Parameters
    ----------
    data : DataFrame, shape (n_samples, >= 1 + n_predictors + n_covariates)
        Contain at least all the potential predictors, the potential covariates and the response 
        
    response : string
        name of the response variable we want to predict.
        
    predictors : list of strings
        names of the potential predictors.
        
    covariates : list of strings
        names of the potential covariates (can be an empty list)

    feat_selector : model object
        feature selector tool with support_ attribute (i.e a list of the names of features
        selected as relevant). For instance see https://github.com/scikit-learn-contrib/boruta_py
    
    Note
    -------  
    response, predictors and covariates must be contained in data.columns
    
    Returns
    -------
    output : Serie, shape (n_predictors)
        for each potential predictor, the associated value is 1 if the predictor is considered relevant 
        for the prediction and 0 otherwise.

    """
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

def _score(row , regressor , data) :
    """ Technical function that we will iterate over the rows of BoShapNet.selections_ in 
    order to score each selection of relevant predictors (see fit method of BoshapNet)
    
    Note
    -------    
    Here we suppose that the regressor has a oob_score
    
    """   
    if row.sum() > 0:
        X , y = data.loc[: , row == 1] , data[row.name]
        reg = regressor.fit(X, y)
        results = [r2_score(y , reg.predict(X)) , reg.oob_score_]
    else:
        results = [np.nan , np.nan]
    return pd.Series(results, index = ['R2_train' , 'R2_OOB'])


def _score_test(row , regressor , data , data_test) : 
    """ Same function than _score but with an independent test data set (see test method of BoShapNet) 
    """
    
    if row.sum() > 0:
        X_train , y_train = data.loc[: , row == 1] , data[row.name]
        X_test , y_test = data_test.loc[: , row == 1] , data_test[row.name]
        reg = regressor.fit(X_train, y_train)
        results = [r2_score(y_train , reg.predict(X_train)) , r2_score(y_test , reg.predict(X_test))]
    else:
        results = [np.nan , np.nan]
    return pd.Series(results, index = ['R2_train' , 'R2_test'])


class BoShapNet(object):
    """ Associates a set of "predictor" variables and "response" variables by fitting
    an ensemble regression method for each response.    
    
    Notes
    ----------
    
    The method used to select the most relevant predictors is called Boruta, please
    refer to the following link for a short explanation:
    https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
   
    Here are the link of the python package which implementS this method:
        - https://github.com/scikit-learn-contrib/boruta_py
    
        
    Parameters
    ----------
    responses : list of strings
        names of the response variables
        
    predictors: list of strings
        names of potential predictors
        
    covariates: list of strings, optional
        names of potential covariates. Default is []
        
    regressor: ensemble regression method
        for example sklearn.ensemble.RandomForestRegressor()
    
    verbose: int, optional
        control the verbosity: the higher, the more messages. Default is 1.
    
    Attributes
    ----------
    selections_: DataFrame with binary values, shape (n_responses , n_predictors + n_covariates)
        selections_[response i , predictor j ]=1 indicates that the predictor j has been selected
        for the response i
            
    summary_: DataFrame
        gather several information for each regression (response vs predictors). For instance, the
        coefficient R2 for the training set.
    
    graph_: networkx object
    
    """
    
    def __init__(self , regressor , responses , predictors , covariates = [] , n_jobs = 1):
        self.responses = responses
        self.predictors = predictors
        self.covariates = covariates
        
        self.regressor = regressor

        self.n_jobs = n_jobs
        
        self.selections_ = None
        self.summary_ = None
        self.graph_ = None
    
    
    def fit(self , df , alpha = 0.01 , validation_regressor = None):
        """ Fits an ensemble regression model for each response using the dataframe df. Saves the 
        selected features in the dataframe self.selections_        

        Parameters
        ----------
        df : DataFrame
            columns must at least contain the predictors, the responses and the covariates
        
        alpha: float between 0 and 1, optional
            Default is 0.01.
        
        validation_regressor : model object, optional
            regression model with an oob_score (typically RandomForestRegressor(oob_score = True).
            Default is None.

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
        
        self.selections_ = (pd.concat(selection , axis=1).T)[self.responses] 
        
        if not (validation_regressor is None):
            self.summary_ = self.selections_.apply( _score 
                                                   , args = (validation_regressor , df) 
                                                   , axis = 1)
        return
    
    def test(self , regressor , df_train , df_test):
        """ Tests the selected regression model for each response, i.e response ~ relevant predictors
        , using the data sets df_train and df_test. For each response the train and test r2 scores are
        added to the self.summary_ dataframe.
    
        Parameters
        ----------
        
        regressor : model object
            regression model (e.g RandomForestRegressor()) with fit and predict methods.
            
        df_train : DataFrame
            columns must at least contain the predictors, the responses and the covariates
            
        df_test : DataFrame
            columns must at least contain the predictors, the responses and the covariates

        Returns
        -------
        None.

        """
        if self.summary_ is None : 
            self.summary_ = self.selections_.apply( _score_test 
                                                   , args = (regressor , df_train , df_test) 
                                                   , axis = 1)
        else : 
            self.summary_ = pd.concat([self.summary_ , self.selections_.apply( _score_test , args = (regressor , df_train , df_test) , axis = 1)]
                                       , keys = ['Validation' , 'Test'] , axis = 1)
        return 

    
    def create_graph(self , label_responses = None , label_predictors = None):
        """ Uses the selections_ attribute to build a graph whose nodes are the responses
        and the predictors. Each edge links a response and one of its associated relevant predictor.
        
        Parameters
        ----------
        label_responses : list of strings, optional
            names of the responses. Default is None.
            
        label_predictors : list of strings, optional
            names of the predictors. Default is None.

        Returns
        -------
        dict
           A dictionary with cytoscape json formatted data.

        """
        self.graph_ = nx.Graph()
        for r in self.responses:
            for p in self.predictors:
                if self.selections_.loc[r , p] == 1:
                    self.graph_.add_node(r , label = label_responses)
                    self.graph_.add_node(p , label = label_predictors)
                    self.graph_.add_edge(r , p , weight = 1)
                    
        return nx.readwrite.json_graph.cytoscape_data(self.graph_)
