# === libs ===

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

# === start + run time ===

start = time.time() # run time start
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
                counter += 1 # counter ++
                images = link.findChildren("img", class_="fleft")
                for image in images:
                    bs_output.write(image.get('src'))
                    counter1 += 1 # counter ++
                # print ("Adding", counter, "URL to file...")
                bar() # progress bar ++
        print("Successfully added", counter, "URLs to file.")

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
print(f'Before removing duplicates: {len(imageList)}')  
# print(imageList[0]) # debug 
# print(imageList[1]) # debug 

sortedImageList = list(dict.fromkeys(imageList)) # sort without changing the order
# print(sortedImageList) # debug 
print(f'After removing duplicates: {len(sortedImageList)}') 
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

# === run time ===

# run_time = datetime.now()-start
end = time.time() # run time end 
run_time = round(end-start,2)
print("Script run time:", run_time, "seconds.")