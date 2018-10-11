TempStr = input('请输入带有符号的温度值:')
print(TempStr[0:-1])
if TempStr[-1] in ['f','F']:
    c = (eval(TempStr[0:-1]) - 32) / 1.8
    print('转化后的温度是{:.2f}C'.format(c))
elif TempStr[-1] in ['c','C']:
    f = 1.8 * eval(TempStr[0:-1]) + 32
    print('转化后的温度是{:.2f}F'.format(f))
else:
    print('输入错误')