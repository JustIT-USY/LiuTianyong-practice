#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    char playerinput;//
    int s;
    srand((unsigned)time(NULL));
    s=rand()%25+65;//�����65~90
    printf("������һ����д��ĸ���в�ȡ��\n");

    for(;;)
    {
        scanf("%c",&playerinput);
        getchar();
        if(s>playerinput)
        {
            printf("����Ӵ���������������ĸ�ĺ���Ӵ�������룺\n");
        }else if(s<playerinput)
        {
            printf("����Ӵ���������������ĸ��ǰ��Ӵ�������룺\n");
        }else
        {
            printf("��ϲ��¶��ˣ�����\n");
            break;
        }
    }
    return 0;
}
