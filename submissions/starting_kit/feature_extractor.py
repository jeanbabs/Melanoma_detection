from skimage.transform import resize
import numpy as np


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty + cropy, startx:startx + cropx]


class FeatureExtractor():
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        return X
        
        

    def transform(self, X):
        X = X.values
        X = X.reshape(-1, 400, 400)
        X = np.array([crop_center(image, 300, 300) for image in X])
        X = X.reshape(-1, 300, 300, 1)
        X = np.array([resize(image, (128, 128, 1), mode='reflect', anti_aliasing=True) for image in X])
        X = X.reshape(-1, 128 * 128)
        return X