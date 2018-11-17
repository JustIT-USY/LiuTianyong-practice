import requests
from multiprocessing import Pool
import os
from lxml import etree
import re


def info_img(url,episode):
    res = requests.get(url)
    selector = etree.HTML(res.text)
    img_url = selector.xpath('//*[@id="img"]/@src')
    # 保存
    try:
        photo = requests.get(img_url[-1])
        fp = open('漫画/{}/{}'.format(episode,re.findall('onepunch-man-(\d+.jpg)',img_url[-1])[-1]), 'wb')
        fp.write(photo.content)
        fp.close()
    except:
        print(url + '异常')

def info(url):
    # 从url得到该剧集数
    episode = re.findall('https://www.mangapanda.com/onepunch-man/(\d+)',url)[-1]
    # 创建该剧集目录
    os.makedirs('漫画/{}'.format(episode))
    res = requests.get(url)
    selector = etree.HTML(res.text)
    # 先xpath 后正则得到该剧集的图片数
    pageSize = selector.xpath('//*[@id="selectpage"]/text()')
    pageSize = re.findall('of (\d+)',pageSize[-1])[-1]
    # 生成该图片url列表
    urls = [url + '/{}' .format(i) for i in range(1, int(pageSize) + 1)]
    # 逐个处理
    for url in urls:
        info_img(url,episode)

if __name__ == '__main__':
    url = 'https://www.mangapanda.com/onepunch-man?tdsourcetag=s_pcqq_aiomsg'
    res = requests.get(url)
    selector = etree.HTML(res.text)
    # 获取该漫画更新进度 正则
    page_max = selector.xpath('//*[@id="latestchapters"]/ul/li[1]/a/@href')[-1]
    page_max = re.findall('/onepunch-man/(\d+)',page_max)[-1]
    # 生成url列表
    urls = ['https://www.mangapanda.com/onepunch-man/{}'.format(i) for i in range(1, int(page_max) + 1)]

    # 开辟线程池 16线程
    pool = Pool(processes=16)
    pool.map(info, urls)
