from cv2 import imread
import os
import Image
import base64
import json
class Recommendations:
    def sendDataToHomePage(self):
        with open("image_1.thumbnail", "rb") as imageFile:
            encoded_image_1 = base64.b64encode(imageFile.read())
            
        with open("image_2.thumbnail", "rb") as imageFile:
            encoded_image_2 = base64.b64encode(imageFile.read())
            
        with open("image_4.thumbnail", "rb") as imageFile:
            encoded_image_3 = base64.b64encode(imageFile.read())
            
        with open("image_5.thumbnail", "rb") as imageFile:
            encoded_image_4 = base64.b64encode(imageFile.read())
         
        data = []
        recommendation_1 = {
                            "thumbnail": encoded_image_1,
                             "location": {
                                          "Lognitude": -34.343, 
                                          "Latitiude": 65.4564
                                          },
                             "description": "The Leaning Tower of Pisa or simply the Tower of Pisa is the campanile, or freestanding bell tower, of the cathedral of the Italian city of Pisa, known worldwide for its unintended tilt.",
                             "title": "Leaning Tower of Pisa",    
                             "key": "14:56",  
                            }
         
        recommendation_2 = {
                            "thumbnail": encoded_image_2,
                             "location": {
                                          "Lognitude": -23.634,
                                          "Latitiude": 64.456
                                          },
                             "description": "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in the Indian city of Agra. It was commissioned in 1632 by the Mughal emperor, Shah Jahan, to house the tomb of his favourite wife, Mumtaz Mahal.",
                             "title": "Taj Mahal",    
                             "key": "#14:34",  
                            }
         
        recommendation_3 = {
                            "thumbnail": encoded_image_3,
                             "location": {
                                          "Lognitude": -53.3463, 
                                          "Latitiude": 56.7657
                                          },
                             "description": "The Colosseum or Coliseum, also known as the Flavian Amphitheatre, is an oval amphitheatre in the centre of the city of Rome, Italy. ",
                             "title": "Colosseum",    
                             "key": "#14:65",  
                            }
         
        recommendation_4 = {
                            "thumbnail": encoded_image_4,
                             "location": {
                                          "Lognitude": -64.3453, 
                                          "Latitiude": 87.454
                                          },
                             "description": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China to protect the Chinese states and empires against the raids and invasions of the various nomadic groups of the Eurasian Steppe.",
                             "title": "Great Wall of China",    
                             "key": "#14:78",  
                            }
        
        data.append(recommendation_1)
        data.append(recommendation_2)
        data.append(recommendation_3)
        data.append(recommendation_4)
        
        json_data = json.dumps(data)
        
        return json_data
        
    def compressImages(self):
        size = 256, 192
        path = 'dataset/game'
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