import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import rampwf as rw
from rampwf.workflows.classifier import Classifier
from rampwf.utils.importing import import_file

# Problem title
problem_title = 'Melanomas detection'

# Prediction type
_target_column_name = 'label'
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------
class FeatureExtractor(object):
    def __init__(self, workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        feature_extractor = import_file(module_path, self.element_names[0])
        fe = feature_extractor.FeatureExtractor()
        fe.fit([X_array[:, :, i] for i in train_is], y_array[train_is])
        return fe

    def test_submission(self, trained_model, X_array):
        fe = trained_model
        X_test_array = fe.transform(X_array)
        return X_test_array
    
class FeatureExtractorClassifier(object):
    """
    Difference with the FeatureExtractorClassifier from ramp-workflow:
    Here we don't want a dataframe for X but a 3d array
    """
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        self.element_names = workflow_element_names
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_array, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, [X_array[:, :, i] for i in train_is])
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return fe, clf

    def test_submission(self, trained_model, X_array):
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_array)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)
        return y_proba
workflow = FeatureExtractorClassifier()

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll')]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=61)
    return cv.split(X, y)

def _read_data(path, typ):
    """
    Read and process data and labels.
    Parameters
    ----------
    path: path to directory that has 'data' subdir
    typ: {'train', 'test'}
    Returns
    -------
    X, y data
    """
    test = os.getenv('RAMP_TEST_MODE', 0)

    suffix = ''
    y_array = []

    try:
        data_path = os.path.join(path, 'data',
                                 'data_{0}{1}.npy'.format(typ, suffix))
        X_array = np.load(data_path, mmap_mode='r')
        # X_array = X_array.reshape(-1, X_array.shape[1] * X_array.shape[2])

        labels_path = os.path.join(path, 'data',
                                   'data_{0}{1}_labels.csv'.format(typ, suffix))
        y_array[:] = np.array(pd.read_csv(labels_path)[_target_column_name])
    except IOError:
        raise IOError("'data/data_{0}.npy' and 'data/labels_{0}.csv' are not "
                      "found. Ensure you ran 'python download_data.py' to "
                      "obtain the train/test data".format(typ))

    if test:
        return X_array[:30], y_array[:30]
    else:
        return X_array, y_array

def get_test_data(path='.'):
    return _read_data(path, 'test')

def get_train_data(path='.'):
    return _read_data(path, 'train')