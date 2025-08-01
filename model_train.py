#!/usr/bin/env python
# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import joblib
import pickle

# function for computing accuracy
def compute_accuracy(Y_true, Y_pred):  
    correctly_predicted = 0  
    for true_label, predicted in zip(Y_true, Y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1  
    return correctly_predicted / len(Y_true)  

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = compute_accuracy(y_test, y_predict)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save model using joblib
joblib.dump(model, 'model.joblib')
