import os
import re
import time
import pandas as pd

import requests
from bs4 import BeautifulSoup

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--query', default='카페')
parser.add_argument('--date_from', default='20190101')
parser.add_argument('--date_to', default='20191231')
parser.add_argument('--num_posts', default=1000)

def load_data(data_path):
    if os.path.isfile(data_path):
        data = pd.read_csv(data_path)
    else:
        data = pd.DataFrame(columns=['title', 'url', 'text'])
    return data

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

def save_data(data, data_path):
    data = data.dropna().reset_index(drop=True)
    data.to_csv(data_path, index=False, encoding='utf-8-sig')
    return data


def main():
    args = parser.parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_dir, 'data', f'{args.query}.csv')
    
    data = load_data(data_path)

    base_url = f'https://search.naver.com/search.naver?where=post&query={args.query}&date_from={args.date_from}&date_to={args.date_to}&date_option=8'
    start = 1
    while True:
        query_url = base_url + f'&start={start}'
        titles, urls = get_post_urls(query_url)
        
        for title, url in zip(titles, urls):
            if url not in data['url'].values:
                mobile_url = convert_to_mobile_version(url)
                text = crawl_text(mobile_url)
                data = data.append({'title':title, 'url':url, 'text':text}, ignore_index=True)
                time.sleep(5)

            if len(data) == args.num_posts:
                break
        
        start += 10
        data = save_data(data, data_path)
        print(f'Collected : {str(len(data)).zfill(4)}')

    data = save_data(data, data_path)
    

if __name__ == '__main__':
    main()