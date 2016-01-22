import cv2
import imutils 
import numpy as np
import os, sys
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans,vq
import pandas
import sklearn_pandas
from sklearn.decomposition import PCA
from sklearn2pmml import sklearn2pmml
import Image



class CreateImageDataSet:
    def ExtractFeatures(self):
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
                im = cv2.imread(image_path)
                kpts = fea_det.detect(im)
                kpts, des = des_ext.compute(im, kpts)
                des_list.append((image_path, des))  
                
            # Stack all the descriptors vertically in a numpy array
            descriptors = des_list[0][1]
            for image_path, descriptor in des_list[1:]:
                descriptors = np.vstack((descriptors, descriptor)) 
                
            self.cluteringDescriptors(descriptors, image_paths, des_list, image_classes, training_name, cluster_dir_names)
            
            
    def cluteringDescriptors(self, descriptors, image_paths, des_list, image_classes, training_name, cluster_dir_names):
        # Perform k-means clustering
        k = 50
        voc, variance = kmeans(descriptors.astype(float), k, 1) 
        
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
        svm_file_name = str(training_name) + ".pkl"
    
        # Save the SVM
        joblib.dump((clf, cluster_dir_names, stdSlr, k, voc), svm_file_name, compress=9)
        

    def compressImages(self):
        size = 256, 192
        path = 'dataset/landmarks'
        dir_names = os.listdir(path)
        for dir_name in dir_names:
            path_to_dir = os.path.join(path, dir_name)
            images = os.listdir(path_to_dir)
            infile = os.path.join(path_to_dir, images[0])
            outfile = os.path.splitext(infile)[0] + ".thumbnail"
            if infile != outfile:
                try:
                    im = Image.open(infile)
                    im.thumbnail(size)
                    im.save(outfile, "JPEG")
                    if(os.path.isfile(infile)):
                        os.remove(infile)
                except IOError:
                    print "cannot create thumbnail for", infile

            
            
            
            
a = CreateImageDataSet()
a.ExtractFeatures()
a.compressImages()