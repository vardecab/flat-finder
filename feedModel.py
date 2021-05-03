# once images are manually reviewed, feed model with them so it can improve itself 

import os 
import shutil 

folderAncient = os.listdir("./images/feeding/ancient")
for image in folderAncient:
    if image.endswith(".jpg"):
        shutil.move("./images/feeding/ancient/" + image, './datasets/flats/ancient/')

folderModern = os.listdir("./images/feeding/modern")
for image in folderModern:
    if image.endswith(".jpg"):
        shutil.move("./images/feeding/modern/" + image, './datasets/flats/modern/')