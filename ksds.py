# -*- coding: utf-8 -*-
import scrapy
from kuaishoudianshang.spiders.ExtendFUNC import Extend
from scrapy import Request
import re
import urllib

class KsdsSpider(scrapy.Spider):
    name = 'ksds'
    allowed_domains = ['kuaishou.com']
    start_urls = ['https://live.kuaishou.com/cate/']

    def parse(self, response):
        category = response.xpath('//*[@id="app"]/div[1]/div[2]/ul/li/div/a[2]/@href').extract()
        category_name = response.xpath('//*[@id="app"]/div[1]/div[2]/ul/li/div/a[2]/p/text()').extract()
        category = Extend.add_list("https://live.kuaishou.com", category)
        for url in category[:2]:
            yield Request(url, callback=self.info_one)
    def info_one(self, response):
        zhubo_Id = response.xpath('//*[@id="app"]/div[1]/div[2]/div[2]/ul/li/div/a/@href').extract()
        zhubo_Id = re.findall("'/u/(.*?)',",str(zhubo_Id))
        for id in zhubo_Id:
            yield Request('https://live.kuaishou.com/profile/'+ id ,callback=self.parse_info)
    def parse_info(self, respoense):
        # 主播名字
        zhubo_titles = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[1]/div[2]/p[1]/text()').extract()[-1]
        # 主播id
        zhubo_id = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[1]/div[2]/p[2]/span[1]/text()').extract()[-1]
        # 主播地区
        zhubo_site = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[1]/div[2]/p[2]/span[2]/text()').extract()[-1]
        # 主播粉丝
        fans = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[2]/div/div[1]/text()').extract()[-1]
        # 自定义解码方法进行处理
        fans = Extend.my_decoding(fans)
        # 主播关注数量
        follow = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[2]/div/div[2]/text()').extract()[-1]
        follow = Extend.my_decoding(follow)
        # 作品数量
        work = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[2]/div/div[3]/text()').extract()[-1]
        work = Extend.my_decoding(work)
        introduction = respoense.xpath('//*[@id="app"]/div[1]/div[2]/div[1]/div[1]/div[2]/p[3]/text()').extract()[-1]
        print(zhubo_id,zhubo_site,zhubo_titles,fans,follow,work,introduction)
