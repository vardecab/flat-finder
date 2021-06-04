# once images are manually reviewed, feed model with them so it can improve itself 

import os
import shutil 

print('Starting...')

folderAncient = os.listdir("./images/feeding/ancient")
for image in folderAncient:
    if image.endswith(".jpg"):
        try:
            shutil.move("./images/feeding/ancient/" + image, './datasets/flats/ancient/')
        except: 
            print("Possible duplicate found, take a look.")
folderModern = os.listdir("./images/feeding/modern")
for image in folderModern:
    if image.endswith(".jpg"):
        try:
            shutil.move("./images/feeding/modern/" + image, './datasets/flats/modern/')
        except: 
            print("Possible duplicate found, take a look.")
print('Done, model fed :)')