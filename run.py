# ==================================== #
#              flat-finder             #
# ==================================== #

# %%
# ------------ import libs ----------- #

import pickle # store data
import sys, os # create new folders or get path to file/folder 
from urllib.request import urlopen # open URLs
from bs4 import BeautifulSoup # BeautifulSoup; parsing HTML
import re # regex; extract substrings
import time # delay execution; calculate script's run time
from datetime import datetime # add IDs to files/folders' names
from alive_progress import alive_bar # progress bar
import webbrowser # open browser or local file/folder 
import ssl # certificate issue workaround: https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
import certifi # certificate issue workaround: https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
from sys import platform # check platform (Windows/Linux/macOS)
if platform == 'win32':
    from win10toast_click import ToastNotifier # Windows 10 notifications
    toaster = ToastNotifier() # initialize win10toast
    # from termcolor import colored # colored input/output in terminal
elif platform == 'darwin':
    import pync # macOS notifications 
import requests # for IFTTT integration to send webhook
import gdshortener # shorten URLs using is.gd 
import wget # download images
import shutil # copy & move files
from random import randint # generate random numbers
import colorama # colored input/output in terminal
from colorama import Fore, Style # colored input/output in terminal
colorama.init(autoreset=True) # if you find yourself repeatedly sending reset sequences to turn off color changes at the end of every print, then init(autoreset=True) will automate that
import pathlib # handle filesystem paths

# %%
# - import TensorFlow & related libs - #
import matplotlib.pyplot as plt
import numpy as np
# import PIL # Pillow, Python Imaging Library; show images
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# %%
# --------- start + run time --------- #

start_time = time.time() # run time start
print(f"{Fore.GREEN}Starting the script...")

# %%
# ---------- train the model --------- #

# %%
# === download/get images to train the model === 

PATH = "./datasets/flats"
data_dir = "./datasets/flats"
data_dir = pathlib.Path(data_dir)

print() # new line 
print(f"{Fore.GREEN}Getting local images from '{data_dir}' to train the model...")

# %%
# === create a dataset from images === 
batch_size = 32
img_height = 180
img_width = 180


# %%
# training dataset
print() # new line 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # 80% of images used for training/learning
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# %%
# validation dataset
print() # new line 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # 80% of images used for training/learning
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# %%
class_names = train_ds.class_names
print() # new line 
print(f"{Fore.GREEN}Classes found: {class_names}")
print() # new line 

# %%
# === configure the dataset for performance ===
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# Dataset.cache() keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# Dataset.prefetch() overlaps data preprocessing and model execution while training.

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# %%
# === standardize the data === 
# The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; in general you should seek to make your input values small. Here, you will standardize values to be in the [0, 1] range by using a Rescaling layer.
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


# %%
# There are two ways to use this layer. You can apply it to the dataset by calling map:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image)) #? 


# %%
# # === create a model ===
num_classes = 5

# %%
# === data augmentation ===
# generating additional training data from your existing examples by augmenting them using random transformations that yield believable-looking images
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# %%
# === dropout === 
# When you apply Dropout to a layer it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.
model = Sequential([
data_augmentation,
layers.experimental.preprocessing.Rescaling(1./255),
layers.Conv2D(16, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Dropout(0.2),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(num_classes)
])

# %%
# === compile and train the model ===
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
# model.summary() # debug 

# %%
# epochs = 2 # test
epochs = 25 #? what's a good number? 20? 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# %%
# === visualize optimised results === 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# calculate how much time it take to train model 
model_time = time.time()

# %%
# ----------- prep the env ----------- #

print() # new line 
print('=== === ===') # separator
print() # new line 

# === have current date & time in exported files' names ===

# https://www.w3schools.com/python/python_datetime.asp
this_run_datetime = datetime.strftime(datetime.now(), '%y%m%d-%H%M%S') # eg 210120-173112

# %%
# === create new folders if they don't exist ===
# output
if not os.path.isdir("output"):
    os.mkdir("output")
    print(f"{Fore.GREEN}Folder created: output")
if not os.path.isdir("output/" + this_run_datetime):
    os.mkdir("output/" + this_run_datetime) # eg 210120-173112
    print(f"{Fore.GREEN}Folder created: output/{this_run_datetime}")
if not os.path.isdir("output/diff"):
    os.mkdir("output/diff")
    print(f"Folder created: output/diff")
# images
if not os.path.isdir("images"):
    os.mkdir("images")
    print(f"{Fore.GREEN}Folder created: images")
if not os.path.isdir("images/" + this_run_datetime):
    os.mkdir("images/" + this_run_datetime) # eg 210120-173112
    print(f"{Fore.GREEN}Folder created: images/{this_run_datetime}")
if not os.path.isdir("images/feeding"):
    os.mkdir("images/feeding") 
    print(f"{Fore.GREEN}Folder created: images/feeding")
if not os.path.isdir("images/feeding/modern"):
    os.mkdir("images/feeding/modern") 
    print(f"{Fore.GREEN}Folder created: images/feeding/modern")
if not os.path.isdir("images/feeding/ancient"):
    os.mkdir("images/feeding/ancient") 
    print(f"{Fore.GREEN}Folder created: images/feeding/ancient")

# %%
# === IFTTT automation === 

try: 
    file_saved_imk = './data/imk.pk'
    with open(file_saved_imk, 'rb') as file:
        ifttt_maker_key = pickle.load(file)
except:
    print("Add file to the folder, otherwise it will fail.")

event_name = 'new-offer' 
webhook_url = f'https://maker.ifttt.com/trigger/{event_name}/with/key/{ifttt_maker_key}'

def run_ifttt_automation(url, date):
    report = {
        "value1": url, 
        "value2": date 
        # "value3": location
    } # data passed to IFTTT
    requests.post(webhook_url, data=report) # send data to IFTTT

# %%
# === open URL in browser ===
def openURL(newOffer):
    if platform != 'darwin': #! FIX: on macOS `%0A/` is added to the end of URL
        try: 
            webbrowser.open(newOffer) # open URL in browser
            time.sleep(5) # wait 5 seconds in-between opening the tabs
        except: 
            pass

# %%
# === pimp Windows 10 notification === 
#* NOTE: unusable right now because there are multiple URLs with offers

# https://stackoverflow.com/questions/63867448/interactive-notification-windows-10-using-python
# def open_url(page_url): 
#     try: 
#         webbrowser.open_new(page_url)
#         print('Opening search results...')  
#     except: 
#         print('Failed to open search results. Unsupported variable type.')

# %%
# === function to scrape data ===

def pullData(page_url, location):

    print() # new line
    print(f"{Fore.GREEN}Location:", location)

    # === shorten the URL === 
    isgd = gdshortener.ISGDShortener() # initialize
    page_url_shortened = isgd.shorten(page_url) # shorten URL; result is in tuple
    print(f"{Fore.GREEN}Page URL:", page_url_shortened[0]) # [0] to get the first element from tuple
    print(Style.RESET_ALL) # colorama - reset output color 
    
    # %%
    pause_duration = 1 # seconds to wait
    print("Waiting for", pause_duration, "seconds before opening URL...")
    with alive_bar(pause_duration, bar="circles", spinner="dots_waves") as bar:
        for _ in range(pause_duration):
            time.sleep(pause_duration) 
            bar()

    # %%
    print("Opening page...")
    # print (page_url) # debug 
    page = urlopen(page_url, context=ssl.create_default_context(cafile=certifi.where())) # certificate issue workaround

    # %%
    print("Scraping page...")
    soup = BeautifulSoup(page, 'html.parser') # parse the page

    # %%
    # 'a' (append) to add lines to existing file vs overwriting
    with open(r"output/" + this_run_datetime + "/1-output.txt", "a", encoding="utf-8") as bs_output:
        # print (colored("Creating local file to store URLs...", 'green')) # colored text on Windows
        
        # %%
        counter = 0 # counter to get # of URLs
        counter1 = 0 # counter to get # of URLs/images
        with alive_bar(bar="classic2", spinner="classic") as bar: # progress bar
            for link in soup.find_all("a", class_="thumb"):
                bs_output.write(link.get('href'))
                images = link.findChildren("img", class_="fleft")
                for image in images:
                    bs_output.write(image.get('src'))
                    counter1 += 1 # counter ++
                # print ("Adding", counter, "URL to file...")
                bar() # progress bar ++
                counter += 1 # counter ++
        print("Successfully added", counter, "offers to the file.")
        print("Successfully added", counter1, "images to the file.")
        print() # new line 
        
# %%
# -------- pagination support -------- #

#* NOTE: works only on OLX 

print() # new line 
print('=== === ===') # separator
print() # new line 

def paginator(page_url, location):
    page = urlopen(page_url, context=ssl.create_default_context(cafile=certifi.where())) # certificate issue workaround; open URL
    soup = BeautifulSoup(page, 'html.parser') # parse the page
    html_content = soup.body.find('a', attrs={'data-cy': 'page-link-last'})
    number_of_pages_to_crawl = re.search('<span>(.*?)</span>', str(html_content)) #* NOTE: auto
    try: # escape if there is only 1 page
        number_of_pages_to_crawl = int(number_of_pages_to_crawl.group(1))
        if number_of_pages_to_crawl >= 3:
            number_of_pages_to_crawl = 3 #* NOTE: force manual; cap the # if there are lots of pages but also makes sure it will work if there is only 1 page
    except AttributeError:
        number_of_pages_to_crawl = 1
    # number_of_pages_to_crawl = 3 #* 
    
    #* NOTE: force manual
    print('===') # separator 
    print() # new line

    print('How many pages are there to crawl?', number_of_pages_to_crawl)

    page_prefix = '?&page='
    page_number = 1 # begin at page=1
    while page_number <= number_of_pages_to_crawl:
        print("Page number:", page_number, "/", number_of_pages_to_crawl) 
        
        full_page_url = f"{page_url}{page_prefix}{page_number}" # add pages to the original URL
        pullData(full_page_url, location) # throw URL to function
        page_number += 1 # go to next page

# %%
# === URLs to scrape ===

page_url_1 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-g%C4%85d%C3%B3w/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_1 = "Gądów"

page_url_2 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-kar%C5%82owice/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_2 = "Karłowice"

page_url_3 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-kleczk%C3%B3w/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_3 = "Kleczków"

page_url_4 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-kozan%C3%B3w/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_4 = "Kozanów"

page_url_5 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-ku%C5%BAniki/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_5 = "Kuźniki"

page_url_6 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-dw%C3%B3r/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_6 = "Nowy Dwór"

page_url_7 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-pilczyce/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_7 = "Pilczyce"

page_url_8 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-popowice/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_8 = "Popowice"

page_url_9 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-r%C3%B3%C5%BCanka/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_9 = "Różanka"

page_url_10 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_10 = "Cały Wrocław"

page_url_11 = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/q-zak%C5%82adowa/?search%5Bfilter_float_price%3Ato%5D=2000&search%5Bfilter_enum_furniture%5D%5B0%5D=yes&search%5Bfilter_float_m%3Afrom%5D=35&search%5Bfilter_enum_rooms%5D%5B0%5D=two&search%5Bfilter_enum_rooms%5D%5B1%5D=three&search%5Bphotos%5D=1&search%5Bprivate_business%5D=private"
location_11 = "Zakładowa"

# throw URLs to function
paginator(page_url_1, location_1)  
paginator(page_url_2, location_2) 
paginator(page_url_3, location_3) 
paginator(page_url_4, location_4) 
paginator(page_url_5, location_5) 
paginator(page_url_6, location_6) 
paginator(page_url_7, location_7) 
paginator(page_url_8, location_8) 
paginator(page_url_9, location_9) 
paginator(page_url_10, location_10) 
paginator(page_url_11, location_11) 

# %%
# === make file more pretty by adding new lines ===

print('=== === ===') # separator

with open(r"output/" + this_run_datetime + "/1-output.txt", "r", encoding="utf-8") as scraping_output_file: # open file...
    print() # new line 
    print("Reading file to clean up...")
    read_scraping_output_file = scraping_output_file.read() # ... and read it

# %%
urls_line_by_line = re.sub(r"#[a-zA-Z0-9]+(?!https$)://|#[a-zA-Z0-9]+|;promoted", "\n", read_scraping_output_file) # add new lines; remove IDs at the end of URL, eg '#e5c6831089'
urls_line_by_line = re.sub(r"461", "461\n", urls_line_by_line) # find & replace to add new lines
urls_line_by_line = re.sub(r"html\?", "html\n", urls_line_by_line) # find & replace to add new lines

# %%
urls_line_by_line = urls_line_by_line.replace("ireland.", "https://ireland.") # make text clickable again
urls_line_by_line = urls_line_by_line.replace("www", "https://www") # make text clickable again
urls_line_by_line = urls_line_by_line.replace("https://https://", "https://") # make text clickable again
urls_line_by_line = urls_line_by_line.replace(";r=270", "") # replace characters to make URL work

# %%
print("Cleaning the file...")

# %%
# === add URLs to list; add new lines so 1 URL = 1 line === 

# %%
imageList = urls_line_by_line.split() # remove "\n"; add to list
# %%
print("File cleaned up. New lines added.")

# add clean lines to file 
with open(r"output/" + this_run_datetime + "/2-clean.txt", "w", encoding="utf-8") as clean_file:
    for element in imageList: # go through the list
        clean_file.write("%s\n" % element) # write to file

# %%
# ---------- download images --------- #

print() # new line 
print('=== === ===') # separator
print() # new line 
print("Downloading images...")
print() # new line 

if platform == 'win32': # Windows
    counter5 = 1 # images (should) start at list[1] 
    with alive_bar(bar="circles", spinner="dots_waves") as bar:
        for image in imageList: # go through the list
            try: 
                imageURL = imageList[counter5] # images (should) start at list[1] 
                try:
                    downloadedImage = wget.download(imageURL, out='images/' + this_run_datetime) # download image
                    counter7 = imageList.index(imageURL) # get item's list index 
                    print(f'Index ID: {counter7}')
                except: # 404
                    pass # ignore the error (most likely 404) and move on
                # print(f'Image downloaded: {downloadedImage}')
                try: 
                    os.rename('images/' + this_run_datetime + '/image', 'images/' + this_run_datetime + '/image' + str(counter7) + '.jpg') # rename files to .jpg
                except: # wrong filename 
                    pass # ignore the error (most likely 404) and move on
                bar() # progress bar
                counter5 += 1 
            except IndexError: # if counter > len(imageList)
                continue

elif platform == 'darwin': # macOS
    ssl._create_default_https_context = ssl._create_unverified_context # disable SSL validation
    # with alive_bar(bar="circles", spinner="dots_waves") as bar: # TODO
    for image in imageList:
        # try: 
        # imageURL = imageList[counter5]
        try:
            downloadedImage = wget.download(image, out='images/' + this_run_datetime) # download image
            counter7 = imageList.index(image) # get item's list index 
            print(f'Index ID: {counter7}')
        except: # 404
            pass # ignore the error (most likely 404) and move on
        # print(f'Image downloaded: {downloadedImage}')
        try: 
            os.rename('images/' + this_run_datetime + '/image', 'images/' + this_run_datetime + '/image' + str(counter7) + '.jpg') # rename files to .jpg
        except: # wrong filename 
            pass # ignore the error (most likely 404) and move on
            # bar()
            # except IndexError: # if counter > len(imageList)
            #     continue

# %%
# === remove files === 
try: 
    os.rename('images/' + this_run_datetime + '/image', 'images/' + this_run_datetime + '/image.html') # rename 'image' file so it can be deleted
except FileNotFoundError:
    print("Can't find 'image' file.")

# remove .html files so we only have .jpg
folderImages = os.listdir('images/' + this_run_datetime)
for website in folderImages:
    if website.endswith(".html"):
        os.remove(os.path.join('images/' + this_run_datetime, website))

# %%
# remove .tmp files from main folder
# folderImages = "images/"
folderMain = os.listdir("./")
for temps in folderMain:
    if temps.endswith(".tmp"):
        os.remove(os.path.join("./", temps))

# %%
# ------- predicting new images ------ #

# %%
print() # new line 
print("Classifying offers - 'ancient' / 'modern'...") 
print() # new line 

path_folderWithImages = 'images/' + this_run_datetime + '/' 
try:
    with os.scandir(path_folderWithImages) as folderWithImages: # 2-20x faster than listdir()
    # for image in os.listdir(path_folderWithImages): #* NOTE: listdir() fragment
        # TODO: progress bar?
        counterModern = 0 # count modern offers
        for image in folderWithImages: # go through list
            randomID = randint(1,1000) # generate random number
        # if image.endswith(".jpg"): #* NOTE: listdir() fragment
            if image.name.endswith(".jpg"):
                # print(os.path.join(path_folderWithImages, image)) #* NOTE: listdir() fragment
                # print(f'Classifying: {image}') #* NOTE: listdir() fragment
                print(f'Classifying: {image.name}')
                # image = os.path.join(path_folderWithImages, image) #* NOTE: listdir() fragment
                imageNumber = re.search('[0-9]+', image.name) # get image ID 
                imageNumber = imageNumber.group(0) # get 1st match
                imageNumber = int(imageNumber) # convert to number
                # print(f'Image ID: {imageNumber}') # debug

                img = keras.preprocessing.image.load_img(image, target_size=(img_height, img_width))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                if class_names[np.argmax(score)] == 'modern':
                    print("modern")
                    print("Accuracy: {:.2f}%".format(100 * np.max(score)))
                    print(f'Offer ID: {imageNumber-1} // Offer URL: {imageList[imageNumber-1]}')
                    print(f'Image ID: {imageNumber} // Image URL: {imageList[imageNumber]}')
                    with open(r"output/" + this_run_datetime + "/3-modern-offers_temp.txt", "a", encoding="utf-8") as modernOffers:
                        modernOffers.write(imageList[imageNumber-1] + "\n")
                        modernOffers.write(imageList[imageNumber] + "\n")

                    # copy images so they can be manually reviewed and used to feed the model
                    shutil.copy2("images/" + this_run_datetime + "/" + image.name, 'images/feeding/modern/' + str(randomID) + "-" + image.name)
                else:
                    print("ancient")
                    print("Accuracy: {:.2f}%".format(100 * np.max(score)))
                    print("Let's move on.")

                    # copy images so they can be manually reviewed and used to feed the model
                    shutil.copy2("images/" + this_run_datetime + "/" + image.name, 'images/feeding/ancient/' + str(randomID) + "-" + image.name)
                print("=== === ===")
                # counter6 += 1

except IndexError: # missing numbers in filenames
    print("No more images to run.")

# remove ireland.apollo URLs (image ones)
bad_words = ['apollo']
try: # "modern" offers are not available / weren't added so let's escape that
    with open(r"output/" + this_run_datetime + "/3-modern-offers_temp.txt", "r", encoding="utf-8") as oldfile, open(r"output/" + this_run_datetime + "/4-modern-offers_temp2.txt", 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)
except FileNotFoundError: # "modern" offers are not available / weren't added so let's escape that
    print(f"{Fore.RED}No modern offers available at this time.") # colorama - red output

# remove duplicates 
try: # "modern" offers are not available so let's escape that
    lines_seen = set() # holds lines already seen
    outfile = open(r"output/" + this_run_datetime + "/5-modern-offers.txt", "w", encoding="utf-8")
    for line in open(r"output/" + this_run_datetime + "/4-modern-offers_temp2.txt", 'r'):
        if line not in lines_seen: # not a duplicate
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
except FileNotFoundError: # "modern" offers are not available so let's escape that
    pass 


# %%
# === compare files === 

try:
    # file_previous_run = open('output/' + previous_run_datetime + '/5-modern-offers.txt', 'r') # 1st file 
    file_previous_run = open('output/diff/' + 'masterfile.txt', 'r') # masterfile 
    file_current_run = open('output/' + this_run_datetime + '/5-modern-offers.txt', 'r') # 2nd file 

    f1 = [x for x in file_previous_run.readlines()] # set with lines from 1st file  
    f2 = [x for x in file_current_run.readlines()] # set with lines from 2nd file 

    diff = [line for line in f1 if line not in f2] # lines present only in 1st file 
    diff1 = [line for line in f2 if line not in f1] # lines present only in 2nd file 
    #* NOTE file2 must be > file1

    if diff1:
        with open('output/diff/diff-' + this_run_datetime + '.txt', 'w') as w:
            counterNewOffers = 1 # counter 
            # with alive_bar(bar="circles", spinner="dots_waves") as bar:
            for newOffer in diff1: # go piece by piece through the differences 
                w.write(newOffer) # write to file
                
                # --------------- IFTTT -------------- #
                # run_ifttt_automation(newOffer, this_run_datetime) # run IFTTT automation with URL
                # print(f"Sending offer {counterNewOffers} to IFTTT...")

                # -------- open URL in browser ------- #
                if counterNewOffers <= 50: # cap new tabs
                    openURL(newOffer) 
                
                # bar() # progress bar ++
                counterNewOffers += 1 # counter++

        if counterNewOffers <= 0: # should not fire 
            print(f"{Fore.YELLOW}\nNo new apartments since last run.") # colorama - yellow output
        else:
            counterNewOffers = counterNewOffers - 1 # script is reading a new line as new offer so let's do a workaround 
            print(f"{Fore.YELLOW}\n{counterNewOffers} new apartment(s) found since last run! Go check them now!") # colorama - yellow output
            if platform == "darwin":
                # pync.notify(f'Nowe mieszkania: {counterNewOffers}', title='flat-finder', open=page_url, contentImage="https://i.postimg.cc/XJskqPGH/apartment.png", sound="Funk") # appIcon="" doesn't work, using contentImage instead
                pync.notify(f'Nowe mieszkania: {counterNewOffers}', title='flat-finder', contentImage="https://i.postimg.cc/XJskqPGH/apartment.png", sound="Funk") # appIcon="" doesn't work, using contentImage instead
            elif platform == "win32":
                # toaster.show_toast(title="flat-finder", msg=f'Nowe mieszkania: {counterNewOffers}', icon_path="../../icons/apartment.png", duration=None, threaded=True, callback_on_click=open_url(page_url)) # duration=None - leave notification in Notification Center; threaded=True - rest of the script will be allowed to be executed while the notification is still active
                toaster.show_toast(title="flat-finder", msg=f'Nowe mieszkania: {counterNewOffers}', icon_path="icons/apartment.ico", duration=None, threaded=True) # duration=None - leave notification in Notification Center; threaded=True - rest of the script will be allowed to be executed while the notification is still active

    else: # check if set is empty - if it is then there are no differences between files 
        print(f"{Fore.GREEN}\nFiles are the same.") # colorama - green output
except IOError: # ugly workaround when diff file doesn't exist but there are some new offers 
    print(f"{Fore.RED}\nNo previous data - can't diff. Either it's a first run, there was no data previously or previous run crashed.") # colorama - red output
    print(Style.RESET_ALL) # colorama - reset output color 
    # with open('output/diff/diff-' + this_run_datetime + '.txt', 'w') as w:
    for newOffer in open(r"output/" + this_run_datetime + "/5-modern-offers.txt", 'r'):
        counterNewOffers = 1 # counter 
            
        # --------------- IFTTT -------------- #
        # run_ifttt_automation(newOffer, this_run_datetime) # run IFTTT automation with URL
        # print(f"Sending offer {counterNewOffers} to IFTTT...")

        # -------- open URL in browser ------- #
        if counterNewOffers <= 50: # cap new tabs
            openURL(newOffer) 

        counterNewOffers += 1 # counter++

# %% 
# ---- print all relevant offers ---- #
try: 
    with open(r"output/diff/diff-" + this_run_datetime + ".txt", "r", encoding="utf-8") as allRelevantOffers:
        offers = allRelevantOffers.read().splitlines() # read & remove new line character
    print() # newline
    for counter9, offer in enumerate(offers):
        print(f"{Fore.GREEN}{offer}") # colorama - green output
except: 
    pass

print(Style.RESET_ALL) # colorama - reset output color 

# %%
# --- open folder for manual review -- #

try:
    print('Opening folders for manual review...')

    pathname = os.path.dirname(sys.argv[0]) # get path to folder containing this script
    thisFolderPath = os.path.abspath(pathname) # get path to folder containing this script
    # print(thisFolderPath) # debug 
    pathAncient = os.path.join(thisFolderPath, 'images/feeding/ancient') # add folder to path
    # print(pathAncient) # debug 
    pathModern = os.path.join(thisFolderPath, 'images/feeding/modern') # add folder to path
    # print(pathModern) # debug 
    webbrowser.open('file:///'+pathAncient) # open 'ancient' folder
    webbrowser.open('file:///'+pathModern) # open 'modern' folder 

    print("Review the images and then run `manualCorrection.py` & `feedModel.py`.\n")
except: 
    print(f"{Fore.RED}Couldn't open the folders.")

# === update masterfile with this run's offers === 

with open(r"output/diff/" + "masterfile.txt", "a", encoding="utf-8") as masterfile:
    for element in imageList: # go through the list
        masterfile.write("%s\n" % element) # write to file

# %%
# === run time ===

end_time = time.time() # run time end 
total_run_time = round(end_time-start_time,2)

dc_time = round(end_time-model_time,2) # 2nd part
model_time = round(model_time-start_time,2) # 1st part

print(f"{Fore.GREEN}Run completed.") # colorama - green output
print(f"Model training: {round(model_time/60,2)} minutes.")
print(f"Downloading images & classifying: {round(dc_time/60,2)} minutes.")
# print(f"Total script run time: {total_run_time} seconds. That's {round(total_run_time/60,2)} minutes.")
print(f"Total script run time: {round(total_run_time/60,2)} minutes.")

print(Style.RESET_ALL) # colorama - reset output color 