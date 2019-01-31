import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, roc_curve 

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType

# Problem title
problem_title = 'Melanomas detection'

# Prediction type
_target_column_name = 'label'
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# Workflow element
workflow = rw.workflows.FeatureExtractorClassifier(
        workflow_element_names=['feature_extractor', 'classifier'])

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

"""
Recall as defined in the starting kit: Sensitivity
"""
class Recall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = recall_score(
            y_true, y_pred, average=None)[1]
        return score
    
"""
Specificity score (or False Positive rate) (Recall of the negative class)
"""
class Specificity(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='spe', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = recall_score(
            y_true, y_pred, average=None)[0]
        return score
"""
1-Complementary of the FPR at threshold yielding 97% TPR
""" 
class SpecificityAtGoodRecall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='spe@97', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(np.argmax(y_true, axis=1), y_pred[:, 0])
        index = np.argmax(tpr >= 0.97)
        return 1 - fpr[index]

"""
Mixed score: weighted average of all the presented metrics
"""
class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision
        self.recall = Recall()
        self.specificity = Specificity()
        self.accuracy = rw.score_types.Accuracy(name='acc')
        self.rocauc = rw.score_types.ROCAUC(name='auc')
        self.spec_good_recall = SpecificityAtGoodRecall()

    def __call__(self, y_true, y_pred):
        hard_true = np.argmax(y_true, axis=1)
        hard_pred = np.argmax(y_pred, axis=1)
        rec = self.recall(hard_true, hard_pred)
        spe = self.specificity(hard_true, hard_pred)
        spegr = self.spec_good_recall(y_true, y_pred)
        acc = self.accuracy(hard_true, hard_pred)
        auc = self.rocauc(y_true, y_pred)
        avg = (rec + spe + acc + 2 * auc + 3 * spegr) / 8
        return 1 - avg
    
score_types = [
    Mixed(),
    SpecificityAtGoodRecall(),
    rw.score_types.ROCAUC(name='auc'),
    Recall(),
    Specificity(),
    rw.score_types.Accuracy(name='acc')]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=61)
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
        # Loading and 2d shaping
        X_array = np.load(data_path, mmap_mode='r')
        X_array = X_array.reshape(-1, X_array.shape[1] * X_array.shape[2])
        # Dataframe conversion
        X_df = pd.DataFrame(data=X_array)
        
        labels_path = os.path.join(path, 'data',
                                   'data_{0}{1}_labels.csv'.format(typ, suffix))
        y_array = np.array(pd.read_csv(labels_path)[_target_column_name])
    except IOError:
        raise IOError("'data/data_{0}.npy' and 'data/labels_{0}.csv' are not "
                      "found. Ensure you ran 'python download_data.py' to "
                      "obtain the train/test data".format(typ))

    if test:
        return pd.concat([X_df[:15], X_df[-15:]]), np.concatenate((y_array[:15],
                        y_array[-15:]))
    else:
        return X_df, y_array

def get_test_data(path='.'):
    return _read_data(path, 'test')

def get_train_data(path='.'):
    return _read_data(path, 'train')