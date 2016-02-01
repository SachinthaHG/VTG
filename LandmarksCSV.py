import csv
from Connector import Connector

class LandmarksCSV:
    def CreateCSV(self):
        writer=csv.writer(open("landmark_locations.csv",'a+'))
        writer.writerow(['Landmark', 'Longitude', 'Latitiude'])
        connection = Connector()
        connection.makeConnection()
        results = connection.retriveLandmarkLocation()

        for i in range(len(results)):
            data_row = []
            data_row.append(results[i].Name)
            data_row.append(results[i].location)
            data_row.append(results[i].location2)
            writer.writerow(data_row)
            
        connection.closeConnection()
        
        
        
a = LandmarksCSV()
a.CreateCSV()


