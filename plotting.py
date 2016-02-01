import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import xlabel
import joblib
# x = [ 13, 14, 9, 20, 2, 19, 6, 7, 16, 5, 1, 3, 5, 25, 17, 2, 18, 10, 0,   8,   6,   5,  17,  26,  14,  14,   5,   0, 11,  11,  10,   2,   3, 17,  23,   5,  26,  10,  12,   3,  22,  10, 0,  10,   5,  21,   0,   3,   9,   0]
# plt.bar(range(0,50), x)
# plt.title('Image represented as a histogram')
# plt.xlabel('clusters')
# plt.ylabel('frequency')
# plt.show()
 
 
km = joblib.load("locations.pkl")
print km
clusterNumber = km.predict([-12.2342, 34.3453])
print clusterNumber
 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
txt_file = r"OpenBCI-RAW-6Hz-60-60s.txt"
csv_file = r"mycsv.csv"
in_txt = csv.reader(open(txt_file, "rb"), delimiter = ',')
out_csv = csv.writer(open(csv_file, 'wb'))
out_csv.writerows(in_txt)
 
data_from_csv = pd.read_csv("mycsv.csv", skipinitialspace=True)
data = data_from_csv.ix[:, 1]
data_to_plot = np.array(data).tolist()
plt.bar(range(0,len(data_to_plot)), data_to_plot)
plt.show()




#
# Step 1: feature engineering
#

# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# 
# import pandas
# import sklearn_pandas
# 
# iris = load_iris()
# 
# iris_df = pandas.concat((pandas.DataFrame(iris.data[:, :], columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]), pandas.DataFrame(iris.target, columns = ["Species"])), axis = 1)
# 
# iris_mapper = sklearn_pandas.DataFrameMapper([
#     (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], PCA(n_components = 3)),
#     ("Species", None)
# ])
# 
# 
# iris = iris_mapper.fit_transform(iris_df)
# # print iris_mapper
# 
# #
# # Step 2: training a logistic regression model
# #
# 
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.svm import SVC
# 
# iris_X = iris[:, 0:3]
# iris_y = iris[:, 3]
# 
# print iris_X
# iris_classifier = SVC(probability=True)
# iris_classifier.fit(iris_X, iris_y)

#
# Step 3: conversion to PMML
#

# from sklearn2pmml import sklearn2pmml
# from sklearn.externals import joblib

# joblib.dump(iris_classifier, "estimator.pkl", compress = 9)

# sklearn2pmml(iris_classifier, iris_mapper, "LogisticRegressionIris.pmml")










# import os, sys
# import Image
# 
# size = 256, 192
# 
# # for infile in sys.argv[1:]:
# infile = "dataset/landmarks/buddha_statue/20151110_113132.jpg"
# outfile = os.path.splitext(infile)[0] + ".thumbnail"
# if infile != outfile:
#     try:
#         im = Image.open(infile)
#         im.thumbnail(size)
#         im.save(outfile, "JPEG")
#     except IOError:
#         print "cannot create thumbnail for", infile






























# import cv2
# import imutils 
# import numpy as np
# import os, sys
# from sklearn.svm import SVC
# from sklearn.externals import joblib
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.vq import kmeans,vq
# import pandas
# import sklearn_pandas
# from sklearn.decomposition import PCA
# from sklearn2pmml import sklearn2pmml
# import Image
# 
# 
# 
# class CreateImageDataSet:
#     def ExtractFeatures(self):
#         train_path = 'dataset_1'
#         training_names = os.listdir(train_path)
#         
#         image_paths = []
#         image_classes = []
#         class_id = 0
#         for training_name in training_names:
#             dir = os.path.join(train_path, training_name)
#             class_path = imutils.imlist(dir)
#             image_paths+=class_path
#             image_classes+=[class_id]*len(class_path)
#             class_id+=1
#         
#         # Create feature extraction and keypoint detector objects
#         fea_det = cv2.FeatureDetector_create("ORB")
#         des_ext = cv2.DescriptorExtractor_create("ORB")
#         
#         # List where all the descriptors are stored
#         des_list = []
#         
#         for image_path in image_paths:
#             im = cv2.imread(image_path)
#             kpts = fea_det.detect(im)
#             kpts, des = des_ext.compute(im, kpts)
#             des_list.append((image_path, des))  
#             
#         # Stack all the descriptors vertically in a numpy array
#         descriptors = des_list[0][1]
#         for image_path, descriptor in des_list[1:]:
#             descriptors = np.vstack((descriptors, descriptor))  
#         
#             
#         self.cluteringDescriptors(descriptors, image_paths, des_list, image_classes, training_name, training_names)
#             
#             
#     def cluteringDescriptors(self, descriptors, image_paths, des_list, image_classes, training_name, training_names):
#         # Perform k-means clustering
#         k = 50
#         voc, variance = kmeans(descriptors.astype(float), k, 1) 
#         
#         # Calculate the histogram of features
#         im_features = np.zeros((len(image_paths), k), "float32")
#         for i in xrange(len(image_paths)):
#             words, distance = vq(des_list[i][1],voc)
#             for w in words:
#                 im_features[i][w] += 1
#         
#         
#         
#         # Scaling the words
#         stdSlr = StandardScaler().fit(im_features)
#         im_features = stdSlr.transform(im_features)
# 
#         # Train the Linear SVM
#         clf = SVC(probability=True)
#         clf.fit(im_features, np.array(image_classes))
# #         svm_file_name = str(training_name) + ".pkl"
#     
#         # Save the SVM
#         joblib.dump((clf, training_names, stdSlr, k, voc), "model.pkl", compress=9)
#         
# 
#    
#             
#             
#             
#             
# a = CreateImageDataSet()
# a.ExtractFeatures()