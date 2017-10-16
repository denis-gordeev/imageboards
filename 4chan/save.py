#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import re
import json, math
import codecs
import os
import csv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import cfscrape

'''
used https://github.com/Anorov/cloudflare-scrape
used https://github.com/emmetio/pyv8-binaries
'''
def download (link, board):
    scraper = cfscrape.create_scraper() # returns a requests.Session object
    text = scraper.get(link).text # => "<!DOCTYPE html><html><head>..."
    threads = re.findall('thread/\d+',text)
    threads = set(threads)
    df = pd.read_csv("messages-"+board+".csv", quotechar = '"', sep = "\t", error_bad_lines=False, quoting=csv.QUOTE_NONE)
    nums = []
    if len(df)>0:
        nums = df["Number"]
        nums = list(nums)
        nums = [int(el) for el in nums if (type(el) == str and len(el) < 11) or type(el)==int]
        nums += [int(el) for el in nums if type(el) == float and not math.isnan(el)] 
    f = open('messages-'+board+'.csv', 'a')
    csv_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for thread in threads:
        url = 'http://a.4cdn.org/'+board+'/'+thread+'.json'
        print(url)
        thread_text = scraper.get(url).text
        parse_thread(thread_text, url, thread, board, csv_writer,nums)
    f.close()
def save_thread(thread_txt, thread_num):

    thread_file.write(thread_txt.encode('utf-8'))
    thread_file.close()

def parse_thread(text, thread_url, thread_number, board, csv_writer, nums):

    j = json.loads(text)
    posts = j["posts"]
    for post in posts:
        if "com" in post and not (post["no"] in nums):
            soup = BeautifulSoup(post["com"], 'html.parser')
            comment = soup.get_text().encode('utf-8')
            comment = re.sub('\t', ' ', comment)
            comment = re.sub('\n', ' ', comment)
            try:
                csv_writer.writerow([u'/'+board, comment, thread_url.encode('utf-8'), post["no"], post["now"].encode('utf-8'), post["name"].encode('utf-8')])
            except:
                pass
#download('http://2-chru.net/bb/')
#while 1:
download("http://boards.4chan.org/b/", u'b') #change
download("http://boards.4chan.org/pol/", u'pol')

