# Get Features of dataset, Split data to train and test set

import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score




def split_data(datafile, split_partion, random_state):
    df = pd.read_csv(datafile)
    X = pd.DataFrame(df, columns = ['Datatype', 'Recipient', 'Condition']).to_numpy()
    y = pd.DataFrame(df, columns = ['Class']).to_numpy()
    train_X,test_X,train_y,test_y = train_test_split( X, y,test_size=split_partion,random_state=random_state)
    print('Split data into train/test split for initialisation.')
    return train_X,test_X,train_y,test_y

def feature_names(datafile):
    df = pd.read_csv(datafile)
    return [column for column in df]



