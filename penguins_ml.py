import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pickle

data = pd.read_csv('penguins.csv')

# Remove the na value
data = data.dropna()

output = data['species']
features = data[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

features = pd.get_dummies(features)

output, uniques = pd.factorize(output)

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = 0.8)

rfc = RandomForestClassifier(random_state = 15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)

score = accuracy_score(y_pred, y_test)
print('Accuracy Prediction Value: ', score)

rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

