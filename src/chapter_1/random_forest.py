import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

iowa_file_path = '../../data/chapter_1/train.csv'
home_data = pd.read_csv(iowa_file_path)
home_data = home_data.dropna(axis=1)

print(home_data.describe())

y = home_data.SalePrice

feature_names = ['Lot Area', 'Year Built', '1st Flr SF', '2nd Flr SF', 'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd']
X = home_data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, preds))
