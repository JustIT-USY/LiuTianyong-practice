from bs4 import BeautifulSoup
import requests
import time                                                     #导入相应的头文件

headers ={                                                       #加入头请求
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
}

def get_info(url):
    wb_data = requests.get(url,headers = headers)
    soup = BeautifulSoup(wb_data.text,'lxml')
    times = soup.select('#rankWrap > div.pc_temp_songlist > ul > li > span.pc_temp_tips_r > span')
    ranks = soup.select('div.pc_temp_songlist > ul > li > span.pc_temp_num ')
    titles = soup.select('#rankWrap > div.pc_temp_songlist > ul > li > a ')

    for time,rank,title in zip(times,ranks,titles):
        data = {
            '排行榜':rank.get_text().strip(),
            '时长':time.get_text().strip(),
            '歌曲名字': title.get_text().split('-')[-1],
            '歌手': title.get_text().split('-')[0],
        }
        print(data)                                            #获取信息并通过字典打印
if __name__ == '__main__':                                #定义程序入口
    urls = ['http://www.kugou.com/yy/rank/home/{}-8888.html'.format(number)for number in range(1,20)]
    for url in urls:
        get_info(url)
        time.sleep(1)
