import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import *
import json
import cv2
import sklearn


def imageSearch(descriptorsJson):
    descriptors = json.loads(descriptorsJson)
    descriptors = descriptors['desc']
 
    descriptors = descriptors.replace('[','').split('],')
    descriptors = [map(float, s.replace(']','').split(',')) for s in descriptors]
    descriptors = np.array(descriptors)
    clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")
       
    test_features = np.zeros((1, k), "float32")
    words, distance = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1
            
    test_features = stdSlr.transform(test_features)
          
    try:
        predictions = [classes_names[i] for i in clf.predict(test_features)]
    except Exception as e:
        print e
                
    descriptionPath = "landmark_descriptions/" + predictions[0]          
    f = open(descriptionPath, 'r')
    description = f.read()
           
    return description