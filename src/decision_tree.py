import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)

    return mae

iowa_file_path = '../data/train.csv'
home_data = pd.read_csv(iowa_file_path)

#print(home_data.columns)

y = home_data.SalePrice

feature_names = ['Lot Area', 'Year Built', '1st Flr SF', '2nd Flr SF', 'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd']
X = home_data[ feature_names ]
#print(X.describe())
#print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean absolute error: %d" %(max_leaf_nodes, mae))

final_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
final_model.fit(X, y)
final_preds = final_model.predict(X)
print(final_preds)
final_mae = mean_absolute_error(y, final_preds)
print(final_mae)
