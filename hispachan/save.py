#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import cfscrape
import re
import json
import codecs
import os
import csv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

'''
used https://github.com/Anorov/cloudflare-scrape
used https://github.com/emmetio/pyv8-binaries
'''
def download (link, board):
    scraper = cfscrape.create_scraper() # returns a requests.Session object
    text = scraper.get(link).text # => "<!DOCTYPE html><html><head>..."
    threads = re.findall('thread/\d+',text)
    threads = set(threads)
    for thread in threads:
        url = 'http://a.4cdn.org/'+board+'/'+thread+'.json'
        print(url)
        thread_text = scraper.get(url).text
        parse_thread(thread_text, url, thread, board)

def save_thread(thread_txt, thread_num):

    thread_file.write(thread_txt.encode('utf-8'))
    thread_file.close()

def parse_thread(text, thread_url, thread_number, board):
    df = pd.read_csv("messages-"+board+".csv", quotechar = '"', sep = "\t")
    nums = []
    if len(df)>0:
        nums = df["Number"].astype(int)
        nums = list(nums)
        nums = [el.item() for el in nums]
    f = open('messages-'+board+'.csv', 'a')
    csv_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    j = json.loads(text)
    posts = j["posts"]
    for post in posts:
        if "com" in post and not (post["no"] in nums):
            soup = BeautifulSoup(post["com"], 'html.parser')
            comment = soup.get_text().encode('utf-8')
            csv_writer.writerow([u'/'+board, comment, thread_url.encode('utf-8'), post["no"], post["now"].encode('utf-8'), post["name"].encode('utf-8')])
    f.close()
#download('http://2-chru.net/bb/')

download("http://www.hispachan.org/g/", u'g') #change
download("http://www.hispachan.org/int/", u'int')
