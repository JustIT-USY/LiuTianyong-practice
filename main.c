#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    char playerinput;//
    int s;
    srand((unsigned)time(NULL));
    s=rand()%25+65;//随机数65~90
    printf("请输入一个大写字母进行猜取：\n");

    for(;;)
    {
        scanf("%c",&playerinput);
        getchar();
        if(s>playerinput)
        {
            printf("不对哟！它在你输入的字母的后面哟！请输入：\n");
        }else if(s<playerinput)
        {
            printf("不对哟！它在你输入的字母的前面哟！请输入：\n");
        }else
        {
            printf("恭喜你猜对了！！！\n");
            break;
        }
    }
    return 0;
}
