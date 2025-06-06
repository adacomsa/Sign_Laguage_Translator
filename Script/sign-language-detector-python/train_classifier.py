import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Standardize data to ensure consistent shape
max_length = max(len(item) for item in data_dict['data'])
data = np.array([np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item[:max_length] for item in data_dict['data']])
labels = np.asarray(data_dict['labels'], dtype=int)  # Replaced np.int with int

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
