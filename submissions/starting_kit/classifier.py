from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        y_pred_proba = self.clf.predict_proba(X)
        return y_pred_proba