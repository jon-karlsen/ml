import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    X = pd.read_csv('../../data/chapter_2/train.csv', index_col='Id') 
    X_test = pd.read_csv('../../data/chapter_2/test.csv', index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
    X_train.drop(cols_with_missing, axis=1, inplace=True)
    X_valid.drop(cols_with_missing, axis=1, inplace=True)

    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    print("Head:")
    print(X_train.head())

    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

    print()
    print("Categorical variables:")
    print(object_cols)

    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    print()
    print("Unique entries by column (ASC):")
    print(sorted(d.items(), key=lambda x: x[1]))

    return (X_train, X_valid, y_train, y_valid, object_cols)
