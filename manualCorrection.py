# move images once manual review is done

import os 
import shutil 

print('Starting...')

folderAncient = os.listdir("./images/feeding/ancient-to-modern/")
for image in folderAncient:
    if image.endswith(".jpg"):
        shutil.move("./images/feeding/ancient-to-modern/" + image, './images/feeding/modern/')

folderModern = os.listdir("./images/feeding/modern-to-ancient/")
for image in folderModern:
    if image.endswith(".jpg"):
        shutil.move("./images/feeding/modern-to-ancient/" + image, './images/feeding/ancient/')

print('Done, images moved :)')