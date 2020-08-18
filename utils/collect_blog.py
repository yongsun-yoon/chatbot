import os
import re
import time
import pandas as pd

import requests
from bs4 import BeautifulSoup

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query', default='호텔')
parser.add_argument('--date_from', default='20190101')
parser.add_argument('--date_to', default='20191231')
parser.add_argument('--num_posts', default=3000)


def url_to_soup(url):
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def get_post_urls(query_url):
    soup = url_to_soup(query_url)
    posts = soup.find_all('a', class_='sh_blog_title')
    titles = [i.get('title') for i in posts]
    urls = [i.get('href') for i in posts]
    
    include = ['blog.naver.com' in i for i in urls]
    titles = [i for i,j in zip(titles, include)]
    urls = [i for i,j in zip(urls, include)]
    return titles, urls


def convert_to_mobile_version(url):
    url = url.split('//')
    url = url[0] + '//m.' + url[1]
    return url

def crawl_text(url):
    try:
        soup = url_to_soup(url)
        text = [i.text for i in soup.find_all('p')]
        text = [re.sub('[^0-9ㄱ-힣]', ' ', i) for i in text]
        text = ' '.join(text)
    except:
        text = None
    return text



def main():
    args = parser.parse_args()

    titles, urls, texts = [], [], []
    base_url = f'https://search.naver.com/search.naver?where=post&query={args.query}&date_from={args.date_from}&date_to={args.date_to}&date_option=8'
    start = 1

    while True:
        query_url = base_url + f'&start={start}'
        post_titles, post_urls = get_post_urls(query_url)
        post_mobile_urls = [convert_to_mobile_version(i) for i in post_urls]
        
        post_texts = []
        for i in post_mobile_urls:
            post_texts.append(crawl_text(i))
            time.sleep(3)

        titles += post_titles
        urls += post_urls
        texts += post_texts

        start += 10
        if len(titles) > args.num_posts:
            break

    data = pd.DataFrame({'title':titles, 'url':urls, 'text':texts})
    data = data.dropna().drop_duplicates().reset_index(drop=True).iloc[:args.num_posts]

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    save_path = os.path.join(base_dir, 'data', f'{args.query}.csv')
    data.to_csv(save_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()