import requests
from lxml import etree
import time
import random
import csv

write = open('豆瓣电影.csv', 'wt', newline='', encoding='utf-8')
writer = csv.writer(write)
writer.writerow(('电影名', '英文名', '又名', '导演/主演', '年份/国别/类型',
                  '评论人数', '评分',))

def info(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
    }
    res = requests.get(url, headers = headers)
    res.encoding = 'utf-8'
    selector = etree.HTML(res.text)
    # 电影名称
    moveNames = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[1]/a/span[1]/text()')
    # 英文名
    englishNames = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[1]/a/span[2]/text()')
    # 又名
    aliass = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[1]/a/span[3]/text()')
    # 导演
    directors = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[2]/p[1]/text()[1]')
    # 国家
    countrys = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[2]/p[1]/text()[2]')
    # 评论数
    commentNums = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[2]/div/span[4]/text()')
    # 评分
    grades = selector.xpath('//*[@id="content"]/div/div[1]/ol/li/div/div[2]/div[2]/div/span[2]/text()')
    for moveName, englishName, alias, director, country, commentNum, grade in zip(moveNames,englishNames,aliass,directors,countrys,commentNums,grades):
        print(moveName,englishName,alias,director,country,commentNum,grade)
        writer.writerow((moveName, englishName, alias, director, country,
                         commentNum, grade,))
if __name__ == '__main__':
    urls = ['https://movie.douban.com/top250?start={}&filter='.format(i) for i in range(0, 250, 25)]
    for url in urls:
        info(url)
        time.sleep(random.randrange(1,5))