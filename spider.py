import requests
from lxml import etree
import re
import csv

headers ={                                                       #加入头请求
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
}

#产品分类ID 价格 标题 详细描述	图片 多属性组合	多属性主属性
fp = open('ebay商品.csv','a+',newline='',encoding='UTF-8')
writer = csv.writer(fp)
writer.writerow(('商品地址','价格','标题','详细描述','图片','多属性组合','多属性主属性','销量','日期'))

def info(url):
    res = requests.get(url)
    href = res.url
    selector = etree.HTML(res.text)
    # 价格
    price = selector.xpath('//*[@id="prcIsum"]/text()')
    # 标题
    title = selector.xpath('//*[@id="itemTitle"]/text()')
    # 详细内容
    comment = selector.xpath('//*[@id="ds_div"]/p/text()[1]')
    # 图片
    imgs_ = re.findall('<img src="(.*?)" style="max-width:64px;max-height:64px"',res.text,re.S)
    imgs = []
    for i in imgs_:
        imgs.append(i.replace('s-l64','s-l500'))
    # 多属性组合
    label = selector.xpath('//*[@class="vi-bbox-dspn  u-flL lable "]/label/text()')
    label_ = selector.xpath('//*[@class="msku-sel "]/option/text()')
    # 多属性的主属性
    label_main = selector.xpath('//*[@id="mainContent"]/form/div[1]/div[3]/div[1]/label/text()')

    #item = selector.xpath('//*[@id="viTabs_0_is"]/div/table/tbody/tr[2]/td[2]/span/text()')
    # 销量//*[@id="mainContent"]/form/div[1]/div[7]/div[2]/div/span/span[2]/span[2]/a
    sales_volume = selector.xpath('//*[@id="mainContent"]/form/div[1]/div/div[2]/div/span/span/span[2]/a/text()|//*[@id="mainContent"]/form/div[1]/div[3]/div[2]/div/span[2]/span[3]/a/text()')
    sales_volume_url = selector.xpath('//*[@id="mainContent"]/form/div[1]/div/div[2]/div/span/span/span[2]/a/@href|//*[@id="mainContent"]/form/div[1]/div[3]/div[2]/div/span[2]/span[3]/a/@href')
    try:
        res = requests.get(sales_volume_url[-1],headers = headers)
        data = []
        s = re.findall('class="contentValueFont">(.*?)</td>', res.text, re.S)
        for i in range(2, len(s) + 1, 3):
            data.append(s[i])
    except:
        data = []
    if label_.count('- Select -') == 2:
        label[0] = label[0].strip()
        label[-1] = label[-1].strip()

        label_[0]= label[0]
        i = label_.index('- Select -')
        label_[i] = label[1]
        label_main.extend(label_[1:i])
    else:
        try:
            label_[0] = label[-1] + '\n'
        except:
            print("---------------------一个异常---------------")
    writer.writerow((href,price, title, comment, imgs, label_, label_main,sales_volume,data))
    print(href,price, title, comment, imgs, label_, label_main,sales_volume,data)
if __name__ == '__main__':
    key = 'msyeny'
    pgn = 18
    url = 'https://www.ebay.com/str/'+ key+ '?_pgn={}&rt=nc'
    urls = [url.format(i) for i in range(1,pgn + 1)]
    for url in urls:
        res = requests.get(url)
        selector = etree.HTML(res.text)
        urls_ = selector.xpath('//*[@class="s-item "]/div/div[1]/div/a/@href')
        for i in urls_:
            info(i)