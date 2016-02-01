import os
from Connector import Connector
import csv

class createLandmarkClass:
    def insertLandmarks(self):
        connection = Connector()
        connection.makeConnection()
        writer=csv.writer(open("landmark_locations.csv",'a+'))
        train_path = 'dataset/final_train'
        landmark_dir_names = os.listdir(train_path)
        
        for landmark_dir_name in landmark_dir_names:
            dir_path = os.path.join(train_path, landmark_dir_name)
            full_path = str(dir_path) + "/details"
            with open(full_path) as f:
                content = f.read().splitlines()
            connection.insertIntoLandmark(str(landmark_dir_name), str(content[0]), str(content[1]), str(content[2]))
            data_row = []
            data_row.append(landmark_dir_name)
            data_row.append(content[0])
            data_row.append(content[1])
            writer.writerow(data_row)
            
            os.remove(full_path)
            temp_path = str(dir_path) + "/details~"
            if(os.path.isfile(temp_path)):
                os.remove(temp_path)
            
        connection.closeConnection()
        
        


a = createLandmarkClass()
a.insertLandmarks()
