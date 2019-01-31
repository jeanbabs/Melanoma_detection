from sklearn import svm
from sklearn.base import BaseEstimator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Conv2D, Flatten
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        

    def fit(self, X, y):
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        epochs = 25
        batch_size = 32

        self.model.fit(X, y, 
           batch_size=batch_size, 
           epochs=epochs, 
           verbose=0)
        

    def predict_proba(self, X):
        y_pred_proba = self.model.predict_proba(X)
        y_pred_probas = y_pred_proba.tolist()
        for i in y_pred_probas:
            i.append(1-i[0])
        return np.asarray(y_pred_probas)