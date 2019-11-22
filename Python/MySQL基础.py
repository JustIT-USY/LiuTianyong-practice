import pymysql

# 打开数据库连接
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='',db='mydb',charset='UTF8')

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = conn.cursor()

# 使用 execute() 方法执行SQL 查询
cursor.execute("SELECT VERSION()")

# 使用 execute() 方法执行 SQL,如果表存在则删除
cursor.execute("drop table if exists employee")

# 使用预处理语句创建表
sql = """create table employee(
         FirstName char(20) not null,
         LastName char(20),
         Age int,
         Sex char(1),
         Income float )  """
cursor.execute(sql)
print ("\n creat table successd ! \n")


'''   增   '''
# sql 插入语句
sql = """insert into employee(FirstName,
         LastName,Age,Sex,Income) 
         values('Mac','Mohan',20,'M',2000);"""
try:
    # 执行 sql 语句
    cursor.execute(sql)
    # 提交到数据库执行
    conn.commit()
    print ('提交完毕')
except:
    # 如果发生错误则进行回滚
    conn.rollback()
    print('\n Some Error happend ! \n')

# 使用 fetchone() 方法获取单条数据
data = cursor.fetchone()

print ("\n Database version : %s \n"% data)


'''  查   '''
'''
Python查询Mysql使用 fetchone() 方法获取单条数据, 使用fetchall() 方法获取多条数据。
fetchone(): 该方法获取下一个查询结果集。结果集是一个对象
fetchall(): 接收全部的返回结果行. rowcount:
这是一个只读属性，并返回执行execute()方法后影响的行数。
'''

sql = "select * from EMPLOYEE \
       where Income > %d" % (1000)

try:
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in results:
        fname = row[0]
        lname = row[1]
        age = row[2]
        sex = row[3]
        income = row[4]

        # 打印结果
        print ("\n fname =%s,lname =%s,age = %d, sex=%s,income=%d \n "%(fname,lname,age,sex,income))

except:
    print ("Error: unable to fetch data")

'''  改  '''
# SQL 更新语句
sql = "update EMPLOYEE set Age = Age + 1 where sex = '%c'"%('W')

try:
    cursor.execute(sql)
    conn.commit()
    print('Updata successd!')
except:
    conn.rollback()

'''   删   '''
sql = "delete from EMPLOYEE where Age > '%d' "%(50)

try:
    cursor.execute(sql)
    conn.commit()
    print('delete success!')
except:
    conn.rollback()
    print('发生错误，回滚中！')


# 关闭数据库连接
conn.close()
