from sklearn import svm
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = svm.SVC(probability=True, gamma="auto")

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        y_pred_proba = self.clf.predict_proba(X)
        return y_pred_proba