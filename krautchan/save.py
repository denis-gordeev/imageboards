#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import cfscrape
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import csv

def parse_thread(thread_text, url, thread, board):
    df = pd.read_csv("messages-"+board+".csv", quotechar = '"', sep = "\t", error_bad_lines=False)
    nums = []
    if len(df)>0:
        nums = df["Number"]
        nums = list(nums)
	nums = [el.item() for el in nums]
    f = open('messages-'+board+'.csv', 'a')
    csv_writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    soup = BeautifulSoup(thread_text, 'html.parser')
    oppost_comment = soup.find("blockquote").get_text().strip().encode('utf-8')
    oppost_num = soup.find("blockquote").p['id']
    oppost_num = re.sub('post_text_', '', oppost_num)
    oppost_num = int(oppost_num)
    oppost_date = soup.find("span", attrs={"class": "postdate"}).get_text().encode('utf-8')
    oppost_author = soup.find("span", attrs={"class": "postername"}).get_text().encode('utf-8')
    oppost_subject = soup.find("span", attrs={"class": "postsubject"}).get_text().encode('utf-8')
    if len(oppost_comment)>0 and not (oppost_num in nums):
        csv_writer.writerow([u'/'+board, oppost_comment, url.encode('utf-8'), oppost_num, oppost_date, oppost_author, oppost_subject])
    posts = soup.find_all(class_="postreply")
    for post in posts:
        subject = post.find("span", attrs={"class": "postsubject"}).get_text().encode('utf-8')
        author = post.find("span", attrs={"class": "postername"}).get_text().encode('utf-8')
        date = post.find("span", attrs={"class": "postdate"}).get_text().encode('utf-8')
        comment = post.blockquote.get_text().strip().encode('utf-8')
        comment = re.sub('\t', ' ', comment)
        comment = re.sub('\n', ' ', comment)
        num = post.blockquote.p['id']
        num = re.sub('post_text_', '', num)
        num = int(num)
        if len(comment)>0 and not (num in nums):
            csv_writer.writerow([u'/'+board, comment, url.encode('utf-8'), num, date, author, subject])

    f.close()
def download (link, board):
    scraper = cfscrape.create_scraper() # returns a requests.Session object
    text = scraper.get(link).text # => "<!DOCTYPE html><html><head>..."
    threads = re.findall('thread-\d+',text)
    threads = set(threads)
    for thread in threads:
        url = link+thread+".html"
        print(url)
        thread_text = scraper.get(url).text
        parse_thread(thread_text, url, thread, board)
#while 1:
download('http://krautchan.net/p/', 'p')
download('http://krautchan.net/b/', 'b')
