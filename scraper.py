# === import libs ===

import pickle # store data
import os # create new folders
from urllib.request import urlopen # open URLs
from bs4 import BeautifulSoup # BeautifulSoup; parsing HTML
import re # regex; extract substrings
import time # delay execution; calculate script's run time
from datetime import datetime # add IDs to files/folders' names
from alive_progress import alive_bar # progress bar
import webbrowser # open browser
import ssl # certificate issue fix: https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
import certifi # certificate issue fix: https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate
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

# === import TensorFlow & related libs === 
import matplotlib.pyplot as plt
import numpy as np
import PIL # Pillow, Python Imaging Library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# === start + run time ===

start_time = time.time() # run time start
print("Starting...")

# === have current date & time in exported files' names ===

# https://www.w3schools.com/python/python_datetime.asp
this_run_datetime = datetime.strftime(datetime.now(), '%y%m%d-%H%M%S') # eg 210120-173112

file_saved_date = 'data/date.pk'
try: # might crash on first run
    # load your data back to memory so we can save new value; NOTE: b = binary
    with open(file_saved_date, 'rb') as file:
        previous_run_datetime = pickle.load(file) # keep previous_run_datetime (last time the script ran) in a file so we can retrieve it later and compare / diff files 
        print("Previous run:", previous_run_datetime) 
except IOError:
    print("First run - no file exists.") # if it's the first time script is running we won't have the file created so we skip  

try:
    with open(file_saved_date, 'wb') as file: # open pickle file
        pickle.dump(this_run_datetime, file) # dump this_run_datetime (the time script is running) into the file so then we can use it to compare / diff files
        print("This run:", this_run_datetime) 
except IOError:
    print("File doesn't exist.")

# create new folder
if not os.path.isdir("output/" + this_run_datetime):
    os.mkdir("output/" + this_run_datetime) # eg 210120-173112
    print("Folder created:", this_run_datetime)

# === URL to scrape ===

# page_url = "https://www.otodom.pl/wynajem/mieszkanie/wroclaw/"
page_url = "https://www.olx.pl/nieruchomosci/mieszkania/wynajem/wroclaw/"
# TODO: location is used for IFTTT function - can remove
# location = "" 

# === shorten the URL === 

isgd = gdshortener.ISGDShortener() # initialize
page_url_shortened = isgd.shorten(page_url) # shorten URL; result is in tuple
print("Page URL:", page_url_shortened[0]) # [0] to get the first element from tuple

# === IFTTT automation === 

# file_saved_imk = '/data/imk.pk'
# try: # might crash on first run
#     # load your data back to memory so we can save new value; NOTE: b = binary
#     with open(file_saved_imk, 'rb') as file:
#         ifttt_maker_key = pickle.load(file)
# except IOError:
#     print("First run - no file exists.")

# event_name = 'new-car' # TODO: change to new-offer
# webhook_url = f'https://maker.ifttt.com/trigger/{event_name}/with/key/{ifttt_maker_key}'

# def run_ifttt_automation(url, date, location): # TODO: remove location - not needed
#     report = {"value1": url, "value2": date, "value3": location}
#     requests.post(webhook_url, data=report)

# === pimp Windows 10 notification === 

# https://stackoverflow.com/questions/63867448/interactive-notification-windows-10-using-python
def open_url():
    try: 
        webbrowser.open_new(page_url)
        print('Opening search results...')  
    except: 
        print('Failed to open search results. Unsupported variable type.')
    
# === function to scrape data ===

def pullData(page_url):

    # ? can't crawl too often? works better with OTOMOTO limits perhaps
    pause_duration = 2 # seconds to wait
    print("Waiting for", pause_duration, "seconds before opening URL...")
    with alive_bar(pause_duration, bar="circles", spinner="dots_waves") as bar:
        for _ in range(pause_duration):
            time.sleep(1)
            bar()

    print("Opening page...")
    # print (page_url) # debug 
    page = urlopen(page_url, context=ssl.create_default_context(cafile=certifi.where())) # fix certificate issue

    print("Scraping page...")
    soup = BeautifulSoup(page, 'html.parser') # parse the page

    # 'a' (append) to add lines to existing file vs overwriting
    with open(r"output/" + this_run_datetime + "/1-output.txt", "a", encoding="utf-8") as bs_output:
        # print (colored("Creating local file to store URLs...", 'green')) # colored text on Windows
        
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

        # counter = 0 # counter to get # of URLs/images
        # with alive_bar(bar="classic2", spinner="classic") as bar: # progress bar
        #     for link in soup.find_all("img", class_="fleft"):
        #         bs_output.write(link.get('src'))
        #         counter += 1 # counter ++
        #         bar() # progress bar ++
        #         # print ("Adding", counter, "URL to file...")
        # print("Successfully added", counter, "images to file.")

        

# === get the number# of search results pages & run URLs in function ^ ===

# # *NOTE 1/2: perhaps no longer needed as of 0.10? 
# try:
#     open(r"output/" + this_run_datetime + "/1-output.txt",
#          "w").close() # clean main file at start
# except: # crashes on 1st run when file is not yet created
#     print("Nothing to clean, moving on...")
# # *NOTE 2/2: ^

# page = urlopen(page_url, context=ssl.create_default_context(cafile=certifi.where())) # fix certificate issue; open URL
# soup = BeautifulSoup(page, 'html.parser') # parse the page

# number_of_pages_to_crawl = ([item.get_text(strip=True) for item in soup.select("span.page")]) # get page numbers from the bottom of the page
# # print(len(number_of_pages_to_crawl)) # debug; 0 = empty
# if len(number_of_pages_to_crawl) > 0:
#     number_of_pages_to_crawl = int(number_of_pages_to_crawl[-1]) # get the last element from the list ^ to get the the max page # and convert to int 
# else: 
#     number_of_pages_to_crawl = 1
# print('How many pages are there to crawl?', number_of_pages_to_crawl)

# page_prefix = '&page='
# page_number = 1 # begin at page=1
# # for page in range(1, number_of_pages_to_crawl+1):
# while page_number <= number_of_pages_to_crawl:
#     print("Page number:", page_number, "/",
#           number_of_pages_to_crawl) 
#     full_page_url = f"{page_url}{page_prefix}{page_number}"
#     pullData(full_page_url) # throw URL to function
#     page_number += 1 # go to next page

pullData(page_url) # throw URL to function

# === make file more pretty by adding new lines ===

with open(r"output/" + this_run_datetime + "/1-output.txt", "r", encoding="utf-8") as scraping_output_file: # open file...
    print("Reading file to clean up...")
    read_scraping_output_file = scraping_output_file.read() # ... and read it

urls_line_by_line = re.sub(r"#[a-zA-Z0-9]+(?!https$)://|#[a-zA-Z0-9]+|;promoted", "\n", read_scraping_output_file) # add new lines; remove IDs at the end of URL, eg '#e5c6831089'
urls_line_by_line = re.sub(r"461", "461\n", urls_line_by_line) # find & replace
urls_line_by_line = re.sub(r"html\?", "html\n", urls_line_by_line) # find & replace

urls_line_by_line = urls_line_by_line.replace("ireland.", "https://ireland.") # make text clickable again
urls_line_by_line = urls_line_by_line.replace("www", "https://www") # make text clickable again
urls_line_by_line = urls_line_by_line.replace("https://https://", "https://") # make text clickable again

print("Cleaning the file...")

# === remove duplicates & sort === 

imageList = urls_line_by_line.split() # remove "\n"; add to list
# uniqueimageList = list(set(imageList)) # remove duplicates 
# print(f'There are {len(imageList)/2} images in total.') # *NOTE: offers/images

# print(imageList) # debug
# print(f'Before removing duplicates: {len(imageList)}')  
# print(imageList[0]) # debug 
# print(imageList[1]) # debug 

sortedImageList = list(dict.fromkeys(imageList)) # sort without changing the order
# print(sortedImageList) # debug 
# print(f'After removing duplicates: {len(sortedImageList)}') 
# print(sortedImageList[0]) # debug 
# print(sortedImageList[1]) # debug 

print("File cleaned up. New lines added.")

with open(r"output/" + this_run_datetime + "/2-clean.txt", "w", encoding="utf-8") as clean_file:
    # for element in sorted(uniqueimageList): # sort URLs
    # for element in uniqueimageList: # sort URLs
    for element in imageList: # sort URLs
        clean_file.write("%s\n" % element) # write to file

# === tailor the results by using a keyword: brand, model (possibly also engine size etc) === 
# TODO: mostly broken as of 0.9; core works 

# regex_user_input = input("Jak chcesz zawęzić wyniki? Możesz wpisać markę (np. BMW) albo model (np. E39) >>> ") # for now using brand as quesion but user can put any one-word keyword
# regex_user_input = ""
# if len(regex_user_input) == 0:
#     print("Keyword wasn't provided - not searching.")
# else: 
#     regex_user_input = regex_user_input.strip() # strip front & back
#     print("Opening file to search for keyword:", regex_user_input)
#     reg = re.compile(regex_user_input) # matches "KEYWORD" in lines
#     counter2 = 0 # another counter to get the # of search results
#     with open(r'output/' + this_run_datetime + '/3-search_keyword.txt', 'w') as output: # open file for writing
#         print("Searching for keyword...")
#         with open(r'output/' + this_run_datetime + '/2-clean.txt', 'r', encoding='UTF-8') as clean_no_dupes_file: # look for keyword in the clean file without empty lines and duplicates 
#             with alive_bar(bar="circles", spinner="dots_waves") as bar:
#                 for line in clean_no_dupes_file: # read file line by line
#                     if reg.search(line): # if there is a match anywhere in a line
#                         output.write(line) # write the line into the new file
#                         counter2 += 1 # counter ++
#                         bar() # progress bar ++
#                         # print ("Progress:", counter2)
#             if counter2 == 1:
#                 print("Found", counter2, "result.")
#                 # if platform == "win32":
#                 #     toaster.show_toast("otomoto-scraper", "Found " + str(counter2) +
#                 #                        " result.",  icon_path="icons/www.ico", duration=None)
#             else:
#                 print("Found", counter2, "results.")
#                 # if platform == "win32":
#                 #     toaster.show_toast("otomoto-scraper", "Found " + str(counter2) +
#                 #                        " results.",  icon_path="icons/www.ico", duration=None)
# # === open keyword/search results ^ in browser ===

#     if counter2 == 0:
#         print("No search results found.")

#     else:
#         # user_choice_open_urls = input("Chcesz otworzyć linki w przeglądarce? [y/n] >>> ")
#         user_choice_open_urls = 'n'
#         if user_choice_open_urls == 'y':
#             with open("output/" + this_run_datetime + "/3-search_keyword.txt", 'r', encoding='UTF-8') as search_results:
#                 counter3 = 0
#                 print("Opening URLs in browser...")
#                 with alive_bar(bar="circles", spinner="dots_waves") as bar:
#                     for line in search_results: # go through the file
#                         webbrowser.open(line) # open URL in browser
#                         counter3 += 1
#                         bar()
#             if counter3 != 1: # correct grammar for multiple (URLs; them; they)
#                 print("Opened ", str(counter3),
#                     " URLs in the browser. Go and check them before they go 404 ;)")
#                 # if platform == "win32":
#                 #     toaster.show_toast("otomoto-scraper", "Opened " + str(counter3) +
#                 #                        " URLs.",  icon_path="icons/www.ico", duration=None)
#             else: # correct grammar for 1 (URL; it)
#                 print("Opened", counter3,
#                     "URL in the browser. Go and check it before it goes 404 ;)")
#                 # if platform == "win32":
#                 #     toaster.show_toast("otomoto-scraper", "Opened " + str(counter3) +
#                 #                        " URL.",  icon_path="icons/www.ico", duration=None)
#         else:
#             # print ("Ok - URLs saved in 'output/search-output.txt' anyway.")
#             print("Ok - URLs saved to a file.")
#             # print("Script run time:", datetime.now()-start)
#             # sys.exit()

# === download images === 
counter5 = 0 
with alive_bar(bar="circles", spinner="dots_waves") as bar:
    for image in imageList:
        try: 
            imageURL = imageList[counter5]
            try:
                downloadedImage = wget.download(imageURL, out='images/') # download image
            except: # 404
                pass # ignore the error (most likely 404) and move on
            # print(f'Image downloaded: {downloadedImage}')
            try: 
                os.rename('images/image', 'images/image' + str(counter5) + '.jpg') # rename files to .jph
            except: # wrong filename 
                pass # ignore the error (most likely 404) and move on
            bar()
            counter5 += 1 
        except IndexError: # if counter > len(imageList)
            continue

# remove .html files so we only have .jpg
# folderImages = "images/"
folderImages = os.listdir("images/")
for website in folderImages:
    if website.endswith(".html"):
        os.remove(os.path.join("images/", website))

# remove .tmp files from main folder
# folderImages = "images/"
folderMain = os.listdir("./")
for temps in folderMain:
    if temps.endswith(".tmp"):
        os.remove(os.path.join("./", temps))

# === compare files === 

try:
    counter2
except NameError:
    print("Variable not defined. Keyword wasn't provided.") 

    try:
        file_previous_run = open('output/' + previous_run_datetime + '/2-clean.txt', 'r') # 1st file 
        file_current_run = open('output/' + this_run_datetime + '/2-clean.txt', 'r') # 2nd file 

        f1 = [x for x in file_previous_run.readlines()] # set with lines from 1st file  
        f2 = [x for x in file_current_run.readlines()] # set with lines from 2nd file 

        diff = [line for line in f1 if line not in f2] # lines present only in 1st file 
        diff1 = [line for line in f2 if line not in f1] # lines present only in 2nd file 
        # *NOTE file2 must be > file1

        if diff1:
            with open('output/diff/diff-' + this_run_datetime + '.txt', 'w') as w:
                counter4 = 0 # counter 
                with alive_bar(bar="circles", spinner="dots_waves") as bar:
                    for url in diff1: # go piece by piece through the differences 
                        w.write(url) # write to file
                        # run_ifttt_automation(url, this_run_datetime, location) # run IFTTT automation with URL
                        # print('Running IFTTT automation...')
                        bar()
                        counter4 += 1 # counter++
            if counter4 <= 0: # should not fire 
                print ('No new offers since last run.') # *NOTE: offers/images
                # if platform == "darwin":
                #     pync.notify('Nie ma nowych aut.', title='OTOMOTO', open=page_url, contentImage="https://i.postimg.cc/t4qh2n6V/car.png") # appIcon="" doesn't work, using contentImage instead
                # elif platform == "win32":
                #     toaster.show_toast(title="OTOMOTO", msg='Nie ma nowych aut.', icon_path="icons/car.ico", duration=None, threaded=True, callback_on_click=open_url) # duration=None - leave notification in Notification Center; threaded=True - rest of the script will be allowed to be executed while the notification is still active
            else:
                print (counter4, "new offers found since last run! Go check them now!") # *NOTE: offers/images
                # if platform == "darwin":
                #     pync.notify(f'Nowe auta: {counter4}', title='OTOMOTO', open=page_url, contentImage="https://i.postimg.cc/t4qh2n6V/car.png", sound="Funk") # appIcon="" doesn't work, using contentImage instead
                # elif platform == "win32":
                #     toaster.show_toast(title="OTOMOTO", msg=f'Nowe auta: {counter4}', icon_path="../icons/car.ico", duration=None, threaded=True, callback_on_click=open_url) # duration=None - leave notification in Notification Center; threaded=True - rest of the script will be allowed to be executed while the notification is still active
                    # time.sleep(5)
                    # webbrowser.open(page_url)

        else: # check if set is empty - if it is then there are no differences between files 
            print('Files are the same.')
            # if platform == "darwin":
            #         pync.notify('Nie ma nowych aut.', title='OTOMOTO', open=page_url, contentImage="https://i.postimg.cc/t4qh2n6V/car.png") # appIcon="" doesn't work, using contentImage instead
            # elif platform == "win32":
            #     toaster.show_toast(title="OTOMOTO", msg='Nie ma nowych aut.', icon_path="icons/car.ico", duration=None, threaded=True, callback_on_click=open_url) # duration=None - leave notification in Notification Center; threaded=True - rest of the script will be allowed to be executed while the notification is still active
    except IOError:
        print("No previous data - can't diff.")

else:
    print("Keyword was provided; search was successful.") 
    # TODO: same as above but with /[x]-search_keyword.txt

# === model magic ===

# === download/get images to train the model === 
import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
print('Downloading or checking the location...')
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
PATH = './datasets/flats'
# data_dir = tf.keras.utils.get_file(PATH, origin='')
data_dir = './datasets/flats'
data_dir = pathlib.Path(data_dir)
print('Folder location:', data_dir)

# check number of images 
count_check = 230+165 # TODO: automate
image_count = len(list(data_dir.glob('*/*.jpg'))) # !FIX: doesn't work properly
print(f"Image count: {image_count}/{count_check}")


# %%
# test data

## ancient
ancient = list(data_dir.glob('ancient/*'))
PIL.Image.open(str(ancient[0])) # debug


# %%
PIL.Image.open(str(ancient[1])) # debug


# %%
## modern
modern = list(data_dir.glob('modern/*'))
PIL.Image.open(str(modern[0])) # debug


# %%
PIL.Image.open(str(modern[1])) # debug


# %%
# === create a dataset from images === 
batch_size = 32
img_height = 180
img_width = 180


# %%
# training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # 80% of images for validation
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# %%
# validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # 20% of images for validation
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# %%
class_names = train_ds.class_names
print(class_names)


# %%
# === visualize the data === 
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# %%
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break


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
print(np.min(first_image), np.max(first_image))


# %%
# # === create a model ===
num_classes = 5

# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])


# %%
# === compile the model === 
# For this tutorial, choose the optimizers.Adam optimizer and losses.SparseCategoricalCrossentropy loss function. To view training and validation accuracy for each training epoch, pass the metrics argument.
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# %%
# View all the layers of the network using the model's summary method:
# model.summary()


# %%
# === train the model ===
# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# %%
# === visualize training results ===
# accuracy is not great because of overfitting - small number of examples to learn from
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


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
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


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
model.summary()


# %%
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# === save model === 
# Save the entire model as a SavedModel.
# model.save('saved_model/my_model')

# === load model ===
# new_model = tf.keras.models.load_model('saved_model/my_model')

# %%
# === visualize optimised results === 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %%
# === predicting new images ===

# PIL.Image.open(r'#')

counter = 0
PIL.Image.open('images/image' + str(counter) + '.jpg')


# %%
img = keras.preprocessing.image.load_img(
    'images/image' + str(counter) + '.jpg', target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(class_names[np.argmax(score)]) # debug 
if class_names[np.argmax(score)] == 'modern':
    # TODO: magic
    print('YAY')
else:
    # TODO: exit()
    print('NAY')

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
) 

# === run time ===

# run_time = datetime.now()-start
end_time = time.time() # run time end 
run_time = round(end_time-start_time,2)
print("Script run time:", run_time, "seconds.")