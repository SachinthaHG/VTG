import numpy as np
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
import os
import imutils
import cv2
##### cluster data into K=1..10 clusters #####
#K, KM, centroids,D_k,cIdx,dist,avgWithinSS = kmeans.run_kmeans(X,10)

# fName = ('../datasets/UN4col.csv')
# fp = open(fName)
# X = np.loadtxt(fp)
# fp.close()

# fields = ["Longitude", "Latitiude"]
# X = pd.read_csv("landmark_locations.csv", skipinitialspace=True, usecols=fields)
# X = np.array(X)


train_path = 'dataset/trained/clusters'
training_names = os.listdir(train_path)
 
for training_name in training_names:
    image_paths = []
    image_classes = []
    class_id = 0
    full_train_path = os.path.join(train_path, training_name)
    cluster_dir_names = os.listdir(full_train_path)
 
    for cluster_dir_name in cluster_dir_names:
        dirs = os.path.join(full_train_path, cluster_dir_name)
        class_path = imutils.imlist(dirs)
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1
 
    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("ORB")
    des_ext = cv2.DescriptorExtractor_create("ORB")
     
    # List where all the descriptors are stored
    des_list = []
     
    for image_path in image_paths:
#                 print image_path
        im = cv2.imread(image_path)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        print image_path
        print des.shape
        des_list.append((image_path, des))  
         
         
    # Stack all the descriptors vertically in a numpy array
    X = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
#                 print image_path
        X = np.vstack((X, descriptor)).astype(float)
    print X.shape
#     print descriptors
 
 
 
 
 
 
 
 
 
 
 
 
 
 
    K = range(1,10)
      
    # scipy.cluster.vq.kmeans
    KM = [kmeans(X,k) for k in K] # apply kmeans 1 to 10
    centroids = [cent for (cent,var) in KM]   # cluster centroids
      
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
      
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/X.shape[0] for d in dist]  
      
    kIdx = 1
    # plot elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    # ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
    #       markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    tt = plt.title('Elbow for K-Means clustering')  
    plt.show()
    break




























