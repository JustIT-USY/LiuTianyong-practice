import requests
import re
from urllib.parse import unquote
import urllib
import time
import csv

#https://s.2.taobao.com/list/list.htm?spm=2007.1000337.0.0.5da45d40mi3DBy&st_edtime=1&page=3&q=%CA%D6%BB%FA&ist=1



fp = open('咸鱼.csv','wt',newline='',encoding='UTF-8')
writer = csv.writer(fp)
writer.writerow(('名称','地点', '时间','价格','链接',))
def info(url):
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
            'cookie':'UM_distinctid=164b2fb9cae72f-0f23a7df1173c1-4d045769-1fa400-164b2fb9caf657; miid=8898591551160087076; enc=dFRlmysntoAjK8oCTfC0ZniN5c5se0GXe98rangyB6SSUp6cX%2B7XkhrEc8civtlbgTe1tEZqRDzWc8pf2u%2FO2A%3D%3D; thw=cn; cna=mFLKEygOrwECAX1MVtG9fyQ0; hng=CN%7Czh-CN%7CCNY%7C156; v=0; _tb_token_=f53ee75863ee5; unb=3914837091; uc1=cookie16=W5iHLLyFPlMGbLDwA%2BdvAGZqLg%3D%3D&cookie21=V32FPkk%2Fhw%3D%3D&cookie15=W5iHLLyFOGW7aA%3D%3D&existShop=false&pas=0&cookie14=UoTYNkT0PmzfNw%3D%3D&tag=8&lng=zh_CN; sg=918; t=f23fad0498421d7da4a9625345e2482f; _l_g_=Ug%3D%3D; skt=24ce9d808ef4495c; cookie2=182d8a5f9e40611b545f21def9d7f452; cookie1=Vy7sirxy9TJi6ShkKSPjQ4ctotZFsIqXH0iVpSAC7UU%3D; csg=c7cce46d; uc3=vt3=F8dByRmmiEpF5wnsXUE%3D&id2=UNk2TiuSyVgh%2FA%3D%3D&nk2=F5RBwKEfyQzh0pU%3D&lg2=UtASsssmOIJ0bQ%3D%3D; existShop=MTU0MDM2MDc2Mg%3D%3D; tracknick=tb452137379; lgc=tb452137379; _cc_=U%2BGCWk%2F7og%3D%3D; dnk=tb452137379; _nk_=tb452137379; cookie17=UNk2TiuSyVgh%2FA%3D%3D; tg=0; mt=ci=30_1&np=; l=AsTEsO8kaDhFJgALhPLbrwjCHEi33OhH; whl=-1%260%260%261540361071871; x=e%3D1%26p%3D*%26s%3D0%26c%3D0%26f%3D0%26g%3D0%26t%3D0%26__ll%3D-1%26_ato%3D0; isg=BJWVwYxcYgSIg0b4jd2F3p2AqpFFlgWBv7YnXxc664xbbrVg3-BEdSEgPDD99WFc'
    }
    res = requests.get(url, headers = headers)
    sites = re.findall('"provcity":"(.*?)",',res.text)
    titles = re.findall('"describe":"(.*?)",',res.text,re.S)
    hrefs = re.findall('"commentUrl":"(.*?)",',res.text)
    times = re.findall('"publishTime":"(.*?)",',res.text,re.S)
    prices = re.findall('"price":"(.*?)",',res.text)
    return titles,hrefs,times,prices,sites
    # for title, time, price, href, site in zip(titles, times, prices, hrefs, sites):
    #     writer.writerow((title, site, time, price, href,))
def M_main(key):
    # key = '手机'
    # key = urllib.parse.quote(key.encode('gbk'))
    # # %CA%D6%BB%FA
    # #https://s.2.taobao.com/list/list.htm?spm=2007.1000337.0.0.5da45d40mi3DBy&st_edtime=1&page=3&q=%CA%D6%BB%FA&ist=1
    # url = 'https://s.2.taobao.com/list/waterfall/waterfall.htm?wp=3&_ksTS=1540361657256_298&callback=jsonp299&stype=1&st_trust=1&q='+key+'&ist=1'
    key = urllib.parse.quote(key.encode('gbk'))
    url_h = 'https://s.2.taobao.com/list/waterfall/waterfall.htm?wp=3&_ksTS=1540361657256_298&callback=jsonp299&stype=1&st_trust=1&q='
    url = url_h + key + '&ist=1'
    titles, hrefs, times, prices, sites = info(url)
    return titles,hrefs,times,prices,sites

