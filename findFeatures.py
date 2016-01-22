import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from cv2 import resize
import sklearn
import numpy as np
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# Get the training classes names and store them in a list
train_path = 'dataset/testing/test'
training_names = os.listdir(train_path)

for training_name in training_names:
    image_paths = []
    image_classes = []
    class_id = 0
    full_train_path = os.path.join(train_path, training_name)
    cluster_dir_names = os.listdir(full_train_path)

    for cluster_dir_name in cluster_dir_names:
        dir = os.path.join(full_train_path, cluster_dir_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("ORB")
    des_ext = cv2.DescriptorExtractor_create("ORB")
    
    # List where all the descriptors are stored
    des_list = []
    
    for image_path in image_paths:
        print image_path
        im = cv2.imread(image_path)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((image_path, des))  
        
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  

    # Perform k-means clustering
    k = 50
    voc, variance = kmeans(descriptors, k, 1) 
    
    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    
    
    
    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    
    # Train the Linear SVM
    clf = SVC(probability=True)
    clf.fit(im_features, np.array(image_classes))
    svm_file_name = str(training_name) + ".plk"

    # Save the SVM
    joblib.dump((clf, cluster_dir_names, stdSlr, k, voc), svm_file_name, compress=3)   