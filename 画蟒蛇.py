#PythonDraw
#海龟模块
import turtle
turtle.setup(650, 350)        #长，宽， 屏幕中的位置，x，y  后面两个参数可以省略，默认为屏幕中央
turtle.penup()
turtle.fd(-250)
turtle.pendown()
turtle.pensize(25)
turtle.pencolor('yellow')
turtle.seth(-40)
for i in range(4):
    turtle.circle(40, 80)
    turtle.circle(-40, 80)
turtle.circle(40, 80/2)
turtle.fd(40)
turtle.circle(16, 180)
turtle.fd(40 * 2/3)
turtle.done()