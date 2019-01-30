import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import rampwf as rw

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

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll')]

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
        return pd.concat([X_df[:30], X_df[-30:]]), np.concatenate((y_array[:30],
                        y_array[-30:]))
    else:
        return X_df, y_array

def get_test_data(path='.'):
    return _read_data(path, 'test')

def get_train_data(path='.'):
    return _read_data(path, 'train')