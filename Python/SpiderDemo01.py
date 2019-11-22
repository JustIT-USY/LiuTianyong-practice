# -*- coding:UTF-8-*-
'''
@Author : 刘天勇
@Project : Spider
@Email : 1063331689@qq.com
@GitHub : https://github.com/LiuTianyong/SpiderDemo
'''
# 导入相应模块
# time模块主要用于爬虫睡眠
import time
# requests求情页面
import requests

'''三种解析对应的库'''
# BeautifulSoup库解析网页请求
# 返回值类型：Soup文档
# Lxml解析器
# 正则(直接从源码中匹配数据)
from bs4 import BeautifulSoup
from lxml import etree
import re

headers = {
    'User-Agent':'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)'
}

url = 'http://www.kugou.com/yy/rank/home/1-8888.html'
# 请求页面，并得到response
res = requests.get(url,headers=headers)

'''正则匹配'''
titles = re.findall('<li class=" " title="(.*?)" data-index="\d+">',res.text,re.S)
times = re.findall('<span class="pc_temp_time">(.*?)</span>',res.text,re.S)
for title, time in zip(titles, times):
    print(title.split('-')[-1], time.strip(), title.split('-')[0])
'''lxml解析器'''
# selector  = etree.HTML(res.text)
# titles = selector.xpath('//*[@id="rankWrap"]/div[2]/ul/li/a/text()')
# times = selector.xpath('//*[@id="rankWrap"]/div[2]/ul/li/span[4]/span/text()')
# for title, time in zip(titles, times):
#     print(title.split('-')[-1], time.strip(), title.split('-')[0])

'''
# 解析器：            使用方法            优点          缺点
# Python标准库
# BeautifulSoup(res.text, 'html.parser')
# python的内置标准库执行速度适中，文档容错能力强
# Python3.73 / python3.22前版本中文档容错能力差
# 
# lxml HTML解析器 
# BeautifulSoup(res.text, 'lxml')
# 速度快，文档容错能力强
# 需要安装C语言库
# 
# Lxml XML解析器
# BeautifulSoup(res.text, 'lxml') / BeautifulSoup(res.text, ["lxml","xml"])
# 速度快唯一支持XML的解析器
# 需要c语言库
# 
# html5lib
# BeautifulSoup(res.text, 'html5lib')
# 最好的容错性以浏览器的方式解析文档，生成HTML5格式的文档
# 速度慢，不依赖外部扩展
# 
# 官方推荐lxml
'''

'''BeautifulSoup解析器'''
# soup = BeautifulSoup(res.text, 'html.parser')
#
# # 使用浏览器检查定位信息
# titles = soup.select('#rankWrap > div.pc_temp_songlist > ul > li > a')
# times = soup.select('#rankWrap > div.pc_temp_songlist > ul > li > span.pc_temp_tips_r > span')
#
# for title, time in zip(titles,times):
#     print(title.get_text().split(' - ')[-1], title.get_text().split(' - ')[0], time.get_text().strip())
