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
import csv

class getClass:
    def imageSearch(self, descriptorsJson, locationJson, bearingJson):
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
        
        bearing = json.loads(bearingJson)
        bearing = bearing['bearing']
        bearing = bearing.replace('[', '').split('],')
        bearing = [map(float, s.replace(']','').split(',')) for s in bearing]

#         a = 'temp'
#         b = os.listdir(a)
#         for c in b:
#             d = os.path.join(a,c)
#         img = cv2.imread(d)
#         fea_det = cv2.FeatureDetector_create("ORB")
#         des_ext = cv2.DescriptorExtractor_create("ORB")
#         kpts = fea_det.detect(img)
#         kpts, descriptors = des_ext.compute(img, kpts)
#         location = [6.908844, 79.6454667]
#         location = np.array(location)
         
        clusterNumber = self.findCluster(location.reshape(1,-1))
        file_name = "Cluster_" + str(clusterNumber[0]) + ".pkl"
        clf, classes_names, stdSlr, k, voc = joblib.load(file_name)
        test_features = np.zeros((1, k), "float32")
        words, distance = vq(descriptors, voc)
        
        for w in words:
            test_features[0][w] += 1
                 
        test_features = stdSlr.transform(test_features)

        writer=csv.writer(open("buddha_statue_test_data.csv",'a+'))
        try:
            predictions = [classes_names[i] for i in clf.predict(test_features)]
            prediction_probability = np.amax(clf.predict_proba(test_features))
                
            print str(predictions[0]) + " - " + str(prediction_probability)
            data_row = []
            data_row.append(str(predictions[0]))
            data_row.append(str(prediction_probability))
            writer.writerow(data_row)
        except Exception as e:
            print e
        
        connection = Connector()
        connection.makeConnection()
        
        
        
        if(prediction_probability>=0.8):
            results = connection.retriveLandmarkDescription(predictions[0])
            connection.closeConnection()
            
            data=[]
            
            suggetion_path = "dataset/landmarks/" + str(results[0].Name)
            image_names = os.listdir(suggetion_path)
            full_path = os.path.join(suggetion_path, image_names[0])
            with open(full_path, "rb") as imageFile:
                encoded_image = base64.b64encode(imageFile.read())
            
                   
            data.append({
                         "thumbnail": encoded_image,
                         "location": {
                                      "Lognitude": results[0].location, 
                                      "Latitiude": results[0].location2
                                      },
                         "description": results[0].Description,
                         "title": results[0].Name,    
                         "key": str(results[0].rid),   
                         })
            
            json_data = json.dumps(data)
            print json_data
            return json_data
        else:
            results = connection.getLandmarkSuggestions(clusterNumber[0])
            data = []
            for i in range(len(results)):
                suggetion_path = "dataset/landmarks/" + str(results[i].Name)
                image_names = os.listdir(suggetion_path)
                full_path = os.path.join(suggetion_path, image_names[0])
                with open(full_path, "rb") as imageFile:
                    encoded_image = base64.b64encode(imageFile.read())
                
                data.append({
                             "thumbnail": encoded_image,
                             "location": {
                                          "Lognitude": results[i].location, 
                                          "Latitiude": results[i].location2
                                          },
                             "description": results[i].Description,
                             "title": results[i].Name,    
                             "key": str(results[i].rid),   
                             })
            
            json_data = json.dumps(data)
            print json_data
            connection.closeConnection()
            return json_data       
        
    def findCluster(self, location):
        km = joblib.load("locations.pkl")
        clusterNumber = km.predict(location)
        return clusterNumber
 
 
# a = getClass()
# a.imageSearch()   