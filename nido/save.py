#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import cfscrape
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import csv


def parse_thread(thread_text, url, thread, board):
    df = pd.read_csv("messages-" + board + ".csv", quotechar='"',
                     sep="\t", quoting=csv.QUOTE_MINIMAL)
    nums = []
    if len(df) > 0:
        nums = df["Number"]
        nums = list(nums)
        nums = [el.item() for el in nums]
    f = open('messages-' + board + '.csv', 'a')
    csv_writer = csv.writer(f, delimiter='\t', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
    soup = BeautifulSoup(thread_text, 'html.parser')
    oppost_comment = soup.find("div", attrs={"class": "body"}).get_text().strip().encode('utf-8')
    oppost_num = int(thread[5:-5])
    oppost_date = soup.find("time").text[:19]
    oppost_author = soup.find("span", attrs={"class": "name"}).get_text().encode('utf-8')
    oppost_subject = soup.find("span", attrs={"class": "subject"})
    if oppost_subject:
    	oppost_subject = oppost_subject.get_text().encode('utf-8')
    else:
        oppost_subject = ''
    if len(oppost_comment)>0 and not (oppost_num in nums):
        csv_writer.writerow([u'/'+board, oppost_comment, url.encode('utf-8'), oppost_num, oppost_date, oppost_author, oppost_subject])
    posts = soup.find_all(class_="post reply")
    for post in posts:
        subject = oppost_subject
        author = post.find("span", attrs={"class": "name"}).get_text().encode('utf-8')
        date = soup.find("time").text[:19]
        comment = post.div.text.encode('utf-8')
        num = re.findall('id=\"\d*', str(post.p))[0][4:].encode('utf-8')
        # num = post.find_all('a')[1].text.encode('utf-8')
        # print (comment)
        num = int(num)
        # print (num)
        if len(comment)>0 and not (num in nums):
            csv_writer.writerow([u'/'+board, comment, url.encode('utf-8'), num, date, author, subject])

    f.close()
def download (link, board):
    scraper = cfscrape.create_scraper() # returns a requests.Session object
    text = scraper.get(link).text # => "<!DOCTYPE html><html><head>..."
    threads = re.findall('/res/\d+.html',text)
    threads = set(threads)
    for thread in threads:
        url = 'http://www.nido.org/'+ board + thread
        print(url)
        thread_text = scraper.get(url).text
        parse_thread(thread_text, url, thread, board)
#while 1:
download('http://www.nido.org/cl/index.html', 'cl')
download('http://www.nido.org/b/index.html', 'b')
