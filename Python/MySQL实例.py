import pymysql
import csv


class MysSQL:
    def __init__(self):
        self.conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='', db='mydb', charset='UTF8')
        self.cursor = self.conn.cursor()

    def initializeSQL(self):
        self.cursor.execute("drop table if exists USERS")
        sql = """create table USERS(
                 id char(10),
                 lv int,
                 site char(8),
                 answerNum int,
                 topAnswerNum int,
                 accept float,
                 attention int,
                 fans int,
                 bee int )  """
        self.cursor.execute(sql)
        print("\ncreat table successd ! \n")

    ''' 增 '''
    def increaseDataSet(self):
        id, lv, site, answerNum, \
        topAnswerNum, accept ,\
        attention, fans, bee = increaseData[0], increaseData[1], increaseData[2],\
                               increaseData[3], increaseData[4], increaseData[5],\
                               increaseData[6], increaseData[7], increaseData[8].strip()

        try:
            # print(id, int(lv), site, int(answerNum), int(topAnswerNum), float(accept), int(attention), int(fans),
            #       int(bee))
            sql = """insert into users (id,lv,site,answerNum,topAnswerNum,accept,attention,fans,bee) values('{}', {}, '{}', {}, {}, {}, {}, {}, {});""".format(
                id, lv, site, answerNum, topAnswerNum, accept, attention, fans, bee)
            # 执行 sql 语句
            self.cursor.execute(sql)
            # 提交到数据库执行
            self.conn.commit()
            print('提交一条数据到数据库,success!')
        except:
            self.conn.rollback()
            print('\n Some Error happend ! \n')

    ''' 查 '''
    def seekDataSet(self):
        sql = "select * from users\
               where bee > %d" % (100)
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            for row in results:
                print(row)
        except:
            print("Error: unable to fetch data")

    ''' 删 '''
    def deleteDataSet(self):
        sql = "delete from users where lv > '%d' " % (50)

        try:
            self.cursor.execute(sql)
            self.conn.commit()
            print('delete success!')
        except:
            self.conn.rollback()
            print('发生错误，回滚中！')

    ''' 改 '''
    def updateDataSet(self):
        # SQL 更新语句
        sql = "update users set lv = lv * 10 where site = '%s'" % ('朝阳')

        try:
            self.cursor.execute(sql)
            self.conn.commit()
            print('Updata successd!')
        except:
            self.conn.rollback()

with open('马蜂窝用户.csv', 'r+') as fp:
    dataSet = fp.readlines()
    dataHeader = dataSet[0]
    dataSet = dataSet[1:]
    mydb = MysSQL()
    mydb.initializeSQL()
    for data in dataSet:
        increaseData = data.split(',')
        mydb.increaseDataSet()
    mydb.seekDataSet()
    mydb.updateDataSet()
    mydb.deleteDataSet()


