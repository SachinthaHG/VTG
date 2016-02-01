from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os
import shutil
import errno
from Connector import Connector

class LandmarkClustering:
    def make_sure_path_exists(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    
    def clustering(self):    
        fields = ["Landmark", "Longitude", "Latitiude"]
        landmark_location_data = pd.read_csv("landmark_locations.csv", skipinitialspace=True, usecols=fields)
        features_data = landmark_location_data.ix[:, 1:]
        self.landmarks = landmark_location_data.ix[:, 0]
        features_data = np.array(features_data)
        self.landmarks = np.array(self.landmarks)
           
        km = KMeans(1, init='k-means++') # initialize
          
        km = km.fit(features_data)
        self.clusterList  = km.predict(features_data)
        self.centers = km.cluster_centers_
        
        joblib.dump((km), "locations.pkl", compress=9)
        
        
        self.updateLandmarkClass()
        self.moveImagesIntoClusterFolders()
     
    def getCentroids(self):
        return self.centers
    
    def updateLandmarkClass(self):
        connection = Connector()
        connection.makeConnection()
        connection.updateLandmarkClusterID(self.clusterList, self.landmarks)
        connection.closeConnection()
        
    def moveImagesIntoClusterFolders(self):     
        for x in range(len(self.clusterList)):
            currentPathName = "dataset/final_train/" + str(self.landmarks[x])
            newPathName = "dataset/trained/clusters/Cluster_" + str(self.clusterList[x])
            self.make_sure_path_exists(newPathName)
            #os.rename(currentPathName, newPathName)
            shutil.move(currentPathName, newPathName)
    
    
    
a = LandmarkClustering()
a.clustering()    
    