import pyorient

class Connector:
    def __init__(self):
        self.client = pyorient.OrientDB("localhost", 2424)
    
    def makeConnection(self):
        session_id = self.client.db_open("test", "root", "root")
        
    def closeConnection(self):
        self.client.db_close()
        
    def insertIntoLandmark(self, landmarkName, X, Y, description):
        query = 'INSERT INTO Landmark (Name, location, Description) VALUES ("' + landmarkName + '", { "@type": "d", "@class": "location", "X": ' + X + ', "Y": ' + Y + '}, "'+ description +'")'
        results = self.client.command(query)
        
    def updateLandmarkClusterID(self, clusterList, landmarks):
        for x in range(len(clusterList)):
            query = "UPDATE Landmark SET Cluster_ID=" + str(clusterList[x]) + " WHERE Name='" + str(landmarks[x]) + "'"
            result = self.client.command(query)
        
    def retriveLandmarkDescription(self, predictions):
        query = "SELECT Description FROM Landmark WHERE Name='" + predictions + "'"
        results = self.client.command(query)
        return results
        
    def retriveLandmarkLocation(self):
        query = 'SELECT Name, location.X, location.Y FROM Landmark'
        results = self.client.command(query)
        return results
    
    def getLandmarkSuggestions(self, clusterNumber):
        query = "SELECT Name, @rid, Description FROM Landmark WHERE Cluster_ID=" + str(clusterNumber)
        results = self.client.command(query)
        return results