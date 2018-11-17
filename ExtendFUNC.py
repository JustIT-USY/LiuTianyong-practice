'''
@ 该文件定义一些常用方法
'''
class Extend():
    def add_list(list, element):
        return [ i + element for i in list]

    def add_list(element, list):
        return [element + i for i in list]

    # 对加密数字进行解码处理
    def my_decoding(fans):
        corresponding_table = '껞 뷍 쳚 곭 ꯏ 붪 꿍 쾾 껝 뾭'.split(' ')
        original_character = '0 1 2 3 4 5 6 7 8 9'.split(' ')
        new_fans = ''
        for f in fans:
            if f == '.' or f == 'w':
                new_fans = new_fans + f
            else:
                i = corresponding_table.index(f)
                new_fans = new_fans + original_character[i]
        return new_fans