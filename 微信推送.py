import itchat
import os
import time
import cv2
import 咸鱼网.xianyuSpider as xs

sendMsg = u"{消息助手}：暂时无法回复"
usageMsg = u"使用方法：\n1.运行CMD命令：cmd xxx (xxx为命令)\n" \
           u"-例如关机命令:\ncmd shutdown -s -t 0 \n" \
           u"2.获取当前电脑用户：cap\n3.启用消息助手(默认关闭)：ast\n" \
           u"4.启动爬虫:runSpider key\n5.停止爬虫：stopSpider\n" \
           u"6.关闭消息助手：astc"
flag = 0 #消息助手开关
nowTime = time.localtime()
filename = str(nowTime.tm_mday)+str(nowTime.tm_hour)+str(nowTime.tm_min)+str(nowTime.tm_sec)+".txt"
myfile = open(filename, 'w')

@itchat.msg_register('Text')
def text_reply(msg):
    global flag
    message = msg['Text']
    fromName = msg['FromUserName']
    toName = msg['ToUserName']


    if toName == "filehelper":
        if message[0:9] == "runSpider":
            itchat.send("爬虫已经启动", "filehelper")
            titles, hrefs, times, prices, sites = xs.M_main(message.split(' ')[-1])
            for title, href, time, price,site in zip(titles, hrefs, times, prices, sites):
                itchat.send(title + '\n'+ href + '\n' + time +'\n' + price
                + '\n' + site, "filehelper")
        if message == "cap":
            cap = cv2.VideoCapture(0)
            ret, img = cap.read()
            cv2.imwrite("weixinTemp.jpg", img)
            itchat.send('@img@%s'%u'weixinTemp.jpg', 'filehelper')
            cap.release()
        if message[0:3] == "cmd":
            os.system(message.strip(message[0:4]))
        if message == "ast":
            flag = 1
            itchat.send("消息助手已开启", "filehelper")
        if message == "astc":
            flag = 0
            itchat.send("消息助手已关闭", "filehelper")


    elif flag == 1:
        itchat.send(sendMsg, fromName)
        myfile.write(message)
        myfile.write("\n")
        myfile.flush()

if __name__ == '__main__':
    itchat.auto_login()
    itchat.send(usageMsg, "filehelper")
    itchat.run()