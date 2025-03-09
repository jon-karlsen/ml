import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


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


#
#   Drop categorical variables
#
def drop_categorical_vars(X_train, X_valid, y_train, y_valid):
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    print()
    print("MAE from Approach 1 (Drop categorical variables):")
    print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


#
#   Ordinal encoding
#
def ordinal_encoding(X_train, X_valid, y_train, y_valid, object_cols):
    # Columns that can be safely ordinal encoded
    good_label_cols = [col for col in object_cols if
                       set(X_valid[col]).issubset(set(X_train[col]))]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols)-set(good_label_cols))

    print()
    print('Categorical columns that will be ordinal encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    ordinal_encoder = OrdinalEncoder()
    label_X_train[good_label_cols] = ordinal_encoder.fit_transform(label_X_train[good_label_cols])
    label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])

    print()
    print("MAE from Approach 2 (Ordinal Encoding):")
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


#
#   One-hot encoding
#
def one_hot_encoding(X_train, X_valid, y_train, y_valid, object_cols):
    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
    high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

    print()
    print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
    print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

    # Encoding removed index
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    # Ensure string type for all cols
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    print()
    print("MAE from Approach 3 (One-Hot Encoding):")
    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


(X_train, X_valid, y_train, y_valid, object_cols) = load_data()
drop_categorical_vars(X_train, X_valid, y_train, y_valid)
ordinal_encoding(X_train, X_valid, y_train, y_valid, object_cols)
one_hot_encoding(X_train, X_valid, y_train, y_valid, object_cols)
