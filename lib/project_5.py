from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


def load_data_from_database():
    
    """ This function connects pandas to a remote database to access the madelon
    dataset.  Calling the function provides the remote database with the credentials for access.  The function querries and sorts data and returns a dataframe"""
    
    engine = create_engine('postgresql://dsi:correct horse battery staple@joshuacook.me:5432')
    data_frame = pd.read_sql_table(table_name='madelon', con=engine)
    return data_frame


def make_data_dict(X, y, test_size=0.25, random_state_split=None):
    
    """
    This function splits the user supplied feature matrix (X DataFrame) and target vector (y DataFrame) into train and test portions.
    Function returns a data dictionary that contains the original feature matrix (X), target vector (y).  These
    are further divided in train and test portions, respectively.
    User needs to provide feature matrix, target vector, test size (default=0.25)
    and random state (default=None) when calling the function. 
    """
    
    X_train, \
    X_test,  \
    y_train, \
    y_test = train_test_split(X, y, test_size=test_size, random_state=random_state_split)
    
    return {'X': X,
            'X_test': X_test,
            'X_train': X_train,
            'y': y,
            'y_test': y_test,
            'y_train': y_train
           }


def general_transformer(transformer, data_dictionary):
    
    """
    This function scales/transforms the feature train and test matricies utilizing the transformer (StandardScaler,
    SelectKBest) provided by the user when calling the function.  The user must also provide the data dictionary that 
    contains the data (X_train) and test (X_test) that will be used in the function.  The function fits on the train 
    data (X_train) then transforms train and test data.  The function updates the data dictionary, where the X_train 
    and X_test data are now transformed per the transformer selected.  The data dictionary is also appended with 
    the name if the transformer used.
    """
    
    if 'processes' in data_dictionary.keys():
        data_dictionary['processes'].append(transformer)
    else:
        data_dictionary['processes'] = [transformer]
    
    transformer.fit(data_dictionary['X_train'], data_dictionary['y_train'])
    
    data_dictionary['X_train'] = transformer.transform(data_dictionary['X_train'])
    data_dictionary['X_test'] = transformer.transform(data_dictionary['X_test'])
    
    data_dictionary['transformer'] = transformer
    
    return data_dictionary


def general_model(model, data_dictionary):
    
    """
    This function fits on train data (X_train, y_train) and scores the train and test data using the default 
    score metric for the model chosen.  When calling the function, the user provides the model 
    (LogisticRegression, KNeighborsClassifier, GridSearchCV) that the user wants to fit and score.  The user 
    must also provide the data dictionary that contains the X_train, X_test, y_train and y_test data that 
    will be modelled.  The function returns the user supplied data dictionary, which is appended with the 
    model used and train and test scores.  The score metric is the default for the model chosen - please refer
    to the sklearn documentation on a specific model for more details.
    """
    
    if 'processes' in data_dictionary.keys():
        data_dictionary['processes'].append(model)
    else:
        data_dictionary['processes'] = [model]
    
    model.fit(data_dictionary['X_train'], data_dictionary['y_train'])
    
    data_dictionary['train_score'] = model.score(data_dictionary['X_train'], data_dictionary['y_train'])
    data_dictionary['test_score'] = model.score(data_dictionary['X_test'], data_dictionary['y_test'])
    
    data_dictionary['model'] = model
    
    return data_dictionary