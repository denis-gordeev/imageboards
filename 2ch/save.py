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
    threads = re.findall('thread-\d+',text)
    threads = set(threads)


    # read file
    df = pd.read_csv("messages-"+board+".csv", quotechar = '"', sep = "\t", error_bad_lines=False, quoting=csv.QUOTE_NONE, low_memory=True)
    nums = []
    if len(df)>0:
        nums = df["Number"]
        nums = list(nums)
        nums = [el for el in nums]
    f = open('messages-'+board+'.csv', 'a')
    csv_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for thread in threads:
        thread = thread[7:] #remove the word thread-
        url = link+'res/'+thread+".json"
        print(url)
        thread_text = scraper.get(url).text
        parse_thread(thread_text, url, thread, board,csv_writer,nums)
    f.close()


def save_thread(thread_txt, thread_num):
    thread_file.write(thread_txt.encode('utf-8'))
    thread_file.close()

def parse_thread(text, thread_url, thread_number, board, csv_writer, nums):
    j = json.loads(text)
    posts = j["threads"][0]["posts"]
    for post in posts:
        if len(post["comment"])>0 and not (post["num"] in nums):
            comment = post["comment"].encode('utf-8')
            comment = BeautifulSoup(comment)
            comment = re.sub('<.*?>', '', comment).strip()
            comment = re.sub('\t', ' ', comment)
            comment = re.sub('\n', ' ', comment)
            csv_writer.writerow([u'/'+board, comment, thread_url.encode('utf-8'), post["num"], post["date"].encode('utf-8'), post["name"].encode('utf-8'), post["email"].encode('utf-8')])

#download('http://2-chru.net/bb/')
#while 1:

download("http://2ch.hk/b/", u'b') #change
download("http://2ch.hk/po/", u'po')

