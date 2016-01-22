import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import *
import json
import cv2
import sklearn
import pyorient
from Connector import Connector
import os
import base64

class getClass:
    def imageSearch(self, descriptorsJson, locationJson):
#     def imageSearch(self):
        descriptors = json.loads(descriptorsJson)
        descriptors = descriptors['desc']
        descriptors = descriptors.replace(';', '],[')
        descriptors = descriptors.replace('\n', '')
        descriptors = '[' + descriptors + ']'
        descriptors = descriptors.replace('[','').split('],')
        descriptors = [map(float, s.replace(']','').split(',')) for s in descriptors]
        descriptors = np.array(descriptors)
                   
        location = json.loads(locationJson)
        location = location['loc']
        location = location.replace('[','').split('],')
        location = [map(float, s.replace(']','').split(',')) for s in location]
        location = np.array(location)
#        
#         img = cv2.imread("20151110_113144.jpg")
#         fea_det = cv2.FeatureDetector_create("ORB")
#         des_ext = cv2.DescriptorExtractor_create("ORB")
#         kpts = fea_det.detect(img)
#         kpts, descriptors = des_ext.compute(img, kpts)
#         location = [-2.6456, 51.6346]
#         location = np.array(location)
         
        clusterNumber = self.findCluster(location.reshape(1,-1))
        file_name = "Cluster_" + str(clusterNumber[0]) + ".pkl"
        clf, classes_names, stdSlr, k, voc = joblib.load(file_name)

        test_features = np.zeros((1, k), "float32")
        words, distance = vq(descriptors, voc)
        
        print words
        for w in words:
            test_features[0][w] += 1
                  
        test_features = stdSlr.transform(test_features)
         
        try:
            predictions = [classes_names[i] for i in clf.predict(test_features)]
            prediction_probability = np.amax(clf.predict_proba(test_features))
        except Exception as e:
            print e
        
        connection = Connector()
        connection.makeConnection()
        
        if(prediction_probability>=0.9):
            results = connection.retriveLandmarkDescription(predictions[0])
            description = results[0].Description  
            connection.closeConnection()
            
            return description
        else:
            landmarkList = []
           
            results = connection.getLandmarkSuggestions(clusterNumber[0])
            for i in range(len(results)):
                details = []
                suggetion_path = "dataset/landmarks/" + str(results[i].Name)
                image_names = os.listdir(suggetion_path)
                full_path = os.path.join(suggetion_path, image_names[0])
                with open(full_path, "rb") as imageFile:
                    encoded_image = base64.b64encode(imageFile.read())
                details.append(results[i].Name)
                details.append(results[i].Description)
                details.append(encoded_image)
                
                landmarkList.append(details)
                
            data = {}
            
            for j in range(len(results)):
                key = 'result_' + str(j)
                data[key] = landmarkList[j]
            
            json_data = json.dumps(data)
            connection.closeConnection()
            return json_data
                

        
    def findCluster(self, location):
        km = joblib.load("locations.pkl")
        clusterNumber = km.predict(location)
        return clusterNumber
 
 
# a = getClass()
# a.imageSearch()   