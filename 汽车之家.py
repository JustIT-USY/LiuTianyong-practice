import requests
from lxml import etree
import re
import time
from selenium import webdriver
import numpy as np
import csv

fp = open('汽车.csv','wt',newline='',encoding='UTF-8')
writer = csv.writer(fp)
writer.writerow(('评论者id','车型','购车地点','购买时间','车价格','空间','动力','操控','油耗',
                 '舒适性','外观','内饰','性价比','购车目的','评论内容','评论url'))
fp2 = open('链接.txt','w+',newline='',encoding='utf-8')
driver = webdriver.Chrome()  # 选择谷歌浏览器
driver.maximize_window()  # 浏览器窗口最大化

id = '4487'

# proxies = { "http": "124.230.66.216:24917", "https": "175.174.80.43:21537", }

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'
    }

# def info_comment(url):
#     res =requests.get('https:' + url, headers=headers)
#     text = res.text
#     selector = etree.HTML(res.text)
#     # driver.get('https:' + url)
#     # driver.implicitly_wait(13)
#     # pageSource = driver.page_source
#     # text = pageSource
#     # # 匹配ttf font
#     cmp = re.compile("url\('(//.*.ttf)'\) format\('woff'\)")
#     rst = cmp.findall(text)
#     if len(rst) == 0:
#         time.sleep(20)
#         res = requests.get('https:' + url, headers=headers)
#         text = res.text
#         selector = etree.HTML(res.text)
#         cmp = re.compile("url\('(//.*.ttf)'\) format\('woff'\)")
#         rst = cmp.findall(text)
#     ttf = requests.get("http:" + rst[0], stream=True)
#     with open("autohome.ttf", "wb") as pdf:
#         for chunk in ttf.iter_content(chunk_size=1024):
#             if chunk:
#                 pdf.write(chunk)
#     # 解析字体库font文件
#     font = TTFont('autohome.ttf')
#     uniList = font['cmap'].tables[0].ttFont.getGlyphOrder()
#     utf8List = [eval("'\\u" + uni[3:] + "'").encode("utf-8") for uni in uniList[1:]]
#     wordList = ['一', '七', '三', '上', '下', '不', '中', '档', '比', '油', '泥', '灯', '九',
#                 '了', '二', '五', '低', '保', '光', '八', '公', '六', '养', '内', '冷', '副',
#                 '加', '动', '十', '电', '的', '皮', '盘', '真', '着', '路', '身', '软', '过',
#                 '近', '远', '里', '量', '长', '门', '问', '只', '右', '启', '呢', '味', '和',
#                 '响', '四', '地', '坏', '坐', '外', '多', '大', '好', '孩', '实', '小', '少',
#                 '短', '矮', '硬', '空', '级', '耗', '雨', '音', '高', '左', '开', '当', '很',
#                 '得', '性', '自', '手', '排', '控', '无', '是', '更', '有', '机', '来']
#     note = selector.xpath('//*[@class="text-con"]/text()')
#     for i in range(len(utf8List)):
#         note = note.encode("utf-8").replace(utf8List[i], wordList[i].encode("utf-8")).decode("utf-8")
#     return note
def info(url):
    driver.get(url)
    driver.implicitly_wait(15)
    pageSource = driver.page_source
    selector = etree.HTML(pageSource)
    # 评论者id
    commentIds = selector.xpath('//*[@class="name-text"]/p/a/text()')
    # 车型
    motorcycleTypes = selector.xpath('//*[@class="font-arial"]/text()')
    # 地点
    sites = selector.xpath('//*[@class="c333"]/text()')
    # 购买时间 / 车价格
    times = selector.xpath('//*[@class="font-arial bg-blue"]/text()')
    times_ = []
    prices = []
    if len(times) == 75:
        times = np.array(times).reshape((15,5))
        for time1 in times:
            times_.append(time1[0])
            prices.append(time1[1])
    # 空间  动力 操控 油耗  舒适性 外观 内饰 性价比
    spaces = selector.xpath('//*[@class="font-arial c333"]/text()')
    if len(spaces) == 120:
        spaces = np.array(spaces).reshape((15,8))
    # 购车目的
    purposes = selector.xpath('//*[@class="obje"]/text()')
    # 评论内容
    comments = re.findall('class="text-cont js-textcont\d+">(.*?)</div>',pageSource,re.S)
    hrefs = selector.xpath('//*[@class="allcont border-b-solid"]/a/@href')
    commentUrls = []
    i = 2
    for href in hrefs:
        if i % 2 == 0:
            commentUrls.append('https'+href)
            fp2.writelines('https:'+href+'\n')
        i = i + 1

    for commentId,motorcycleType,site,time,price,space,purpose,comment,commentUrl in zip(
            commentIds,motorcycleTypes,sites,times_,prices,spaces,purposes,comments,commentUrls):
        print(commentId, motorcycleType, site, time, prices, space,
                         purpose, comment,commentUrl)
        writer.writerow((commentId, motorcycleType, site, time, price, space[0],space[1],space[2],
                         space[3],space[4],space[5],space[6],space[7],purpose, comment,commentUrl))
if __name__ == '__main__':
    url = 'https://k.autohome.com.cn/' + id +'/index_{}.html#dataList'
    res = requests.get(url.format(1), headers=headers)
    selector = etree.HTML(res.text)
    num = selector.xpath('//*[@class="page-item-info"]/text()')
    if len(num):
        num = re.findall('共(\d+)页', num[-1])[-1]
        urls = [url.format(i) for i in range(1,int(num))]
        for url in urls:
            info(url)
            time.sleep(13)
    else:
        url = 'https://k.autohome.com.cn/'+id + '/index_1.html#dataList'
        info(url)
