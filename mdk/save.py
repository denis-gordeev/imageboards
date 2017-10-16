import re
import vk
import csv
import pandas as pd


session = vk.Session()
api = vk.API(session)
group = '-57846937'
domain = 'mudakoff'

posts = api.wall.get(domain=domain, count=100)[1:]
total_comments = []
for post in posts:
    post_id = post['id']
    comments = api.wall.getComments(owner_id=group,
                                    post_id=post_id,
                                    need_likes=0, count=100)[1:-1]
    for comment in comments:
        if comment not in total_comments and len(comment) == 5:
            total_comments += comments

df = pd.read_csv("messages.csv", quotechar='"',
                 sep="\t", quoting=csv.QUOTE_MINIMAL)
old_comments = list(df.cid)
f = open('messages.csv', 'a')
csv_writer = csv.writer(f, delimiter='\t', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
for comm in total_comments:
    text = comm['text'].encode('utf-8')
    text = re.sub('/t', ' ', text)
    if comm['cid'] not in old_comments:
        csv_writer.writerow([comm['cid'],
                             comm['date'],
                             comm['from_id'],
                             text,
                             comm['uid']])
