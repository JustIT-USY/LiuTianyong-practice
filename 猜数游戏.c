#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
/*�Խ��װ������Ϸ�ṹ�����Ż�
���ӵ�½ע��ȹ���
*/
/*************************************************
Function:������Ϸ��ǿ��
Description:���Բ����֣�����ĸ
Output:
Return:
Others:
auther:������
*************************************************/
int random_num;//�����ֵ����
int g,z;//��������
int playerinput;
char playerinput2;
void function_case1_1();
void function_case1_2();
void function_case1_3();
void function_case2_2();
void windows_1();
int main()
{   system("color A1");
    //�����ֵ����
    int user,password;
    int input_randomnum;//����������֤��
    int input_user,input_PW;
    int i=0;//ѭ������
    int x,y;//�˵�����
    printf("��ӭ����������Ϸ��\n");
    printf("��ע����Ϸ�˺�\n");
    do{
        srand((unsigned)time(NULL));
        random_num=(rand()*rand())%10000+1;
        printf("��������Ҫע����˻���10λ�����֣���\n");
        scanf("%d",&user);
        printf("������������루10λ�����֣���\n");
        scanf("%d",&password);
        printf("��������֤��%d\t��֤�룺",random_num);
        scanf("%d",&input_randomnum);
        if(input_randomnum!=random_num)
        {
            printf("��֤���������������ע�ᣡ\n");
        }
    }while(input_randomnum!=random_num);
    system("cls");
    printf("����˻��ǣ�%d\n��������ǣ�%d\n���μǣ�����",user,password);
    Sleep(700);
    system("cls");
    system("color A0");
    printf("���¼��\n");
    do{
        srand((unsigned)time(NULL));
        random_num=(rand()*rand())%10000+1;
        if(i==3)
        {
            printf("��Ĵ����Ѿ����꣬��˶���Ϣ��������");
            system("exit");
        }
        i++;
        if(i>1)
        {
            printf("�˻������벻��ȷ������������!\n");
        }

        printf("�˺ţ� ");
        scanf("%d",&input_user);
        printf("���룺 ");
        scanf("%d",&input_PW);
        printf("�����֤����%d\t",random_num);
        scanf("%d",&input_randomnum);

    }while((input_user!=user||input_PW!=password||input_randomnum!=random_num)&&i<4);

    printf("��½�ɹ���");
    Sleep(700);
    system("cls");
    printf("**************************************************************\n");
	printf("************* ��  ӭ  ��  ��  ��  ��  ��  Ϸ *****************\n");
	printf("*****                                                    *****\n");
	printf("*****                                                    *****\n");
	printf("*****            1.��    ��    ��    Ϸ                  *****\n");
	printf("*****                                                    *****\n");
	printf("*****            2.��  ��  ĸ   ��   Ϸ                  *****\n");
	printf("*****                                                    *****\n");
	printf("*****            3.��    ��    ��    Ϸ                  *****\n");
	printf("*****                                                    *****\n");
	printf("*****                                                    *****\n");
	printf("**************************************************************\n");
	printf("**************************************************************\n");
	scanf("%d", &x);
	system("cls");//ϵͳ����
	switch (x)
	{
    case 1:
        system("color B0");
		printf("**************************************************************\n");
		printf("************* ��   ѡ   ��   ��   Ϸ   ��   �� ***************\n");
		printf("*****                                                    *****\n");
		printf("*****                                                    *****\n");
		printf("*****            1.��   ��   ��   ��                     *****\n");
		printf("*****                                                    *****\n");
		printf("*****            2.��   ��   ��   ��                     *****\n");
		printf("*****                                                    *****\n");
		printf("*****            3.��   ��   ��   ��                     *****\n");
		printf("*****                                                    *****\n");
		printf("*****                                                    *****\n");
		printf("**************************************************************\n");
		printf("**************************************************************\n");
		scanf("%d", &y);
		system("cls");
        switch (y)
        {
        case 1://�����Ѷ�
            srand((unsigned)time(NULL));
            random_num=rand()%10+1;
            function_case1_1();
            break;
        case 2://�м��Ѷ�
            srand((unsigned)time(NULL));
            random_num=rand()%100+1;
            function_case1_2();
            break;
        case 3://�߼��Ѷ�
            srand((unsigned)time(NULL));
            random_num=rand()%1000+1;
            function_case1_3();
            break;
        default:
            break;
        }
        break;
    case 2://����ĸ��Ϸ
        system("color 20");
        srand((unsigned)time(NULL));
        random_num=rand()%25+65;//�����65~90
        function_case2_2();
        break;
    default:
        break;
	}
    return 0;
}
void function_case1_1()
{
        for (g=1;;g++)
		{
		    printf("��%d�����룡",g);
			printf("������1~10����������ȡ��\n");
			scanf("%d", &playerinput);
			if (playerinput > random_num)
			{
				printf("����������ִ���Ӵ��\t");
			}
			else if (playerinput < random_num)
			{
				printf("�����������С��Ӵ��\t");
			}
			else
			{
			    system("cls");
			    system("color 40");
				printf("��ϲ��¶��ˣ�����\n");
				if(g==1)
                {
                    printf("�������ѿ��ƺ쳾�����Ѷ��Լ����ʺ����ˣ��������%d��Ӵ��\n",g);
                }else if(1<g<4)
                {
                    printf("���񻹴�Ӵ���������%d��Ӵ��\n",g);
                }else if(3<g<6)
                {
                    printf("������Ӵ�����������%d��Ӵ��\n",g);
                }else
                {
                    printf("������������������ɣ���\n");
                }
                g=0;
				windows_1();
				scanf("%d", &z);
				if (z == 1)
				{
				    system("color B0");
				    srand((unsigned)time(NULL));
				    random_num= rand() % 10;

				}
				else
				{
				    printf("�س�������˳�");
					break;
				}
				continue;
			}
		}
}
void function_case1_2()
{
        for (g=1;;g++)
		{
		    printf("��%d�����룡",g);
			printf("������1~10����������ȡ��\n");
			scanf("%d", &playerinput);
			if (playerinput > random_num)
			{
				printf("����������ִ���Ӵ��\t");
			}
			else if (playerinput < random_num)
			{
				printf("�����������С��Ӵ��\t");
			}
			else
			{
			    system("cls");
			    system("color 40");
				printf("��ϲ��¶��ˣ�����\n");
                if(g==1)
                {
                    printf("���ѳ�Խ����֮�⣡�������%d��Ӵ��\n",g);
                }else if(1<g<11)
                {
                    printf("���񻹴�Ӵ���������%d��Ӵ��\n",g);
                }else if(10<g<30)
                {
                    printf("������Ӵ�����������%d��Ӵ��\n",g);
                }else
                {
                    printf("������������������ɣ���\n",g);
                }
                g=0;
				windows_1();
				scanf("%d", &z);
				if (z == 1)
				{
				    system("color B0");
				    srand((unsigned)time(NULL));
				    random_num= rand() % 100+1;

				}
				else
				{
				    printf("�س�������˳�");
					break;
				}
				continue;
			}
		}
}
void function_case1_3()
{
        for (g=1;;g++)
		{
		    printf("��%d�����룡",g);
			printf("������1~10����������ȡ��\n");
			scanf("%d", &playerinput);
			if (playerinput > random_num)
			{
				printf("����������ִ���Ӵ��\t");
			}
			else if (playerinput < random_num)
			{
				printf("�����������С��Ӵ��\t");
			}
			else
			{
			    system("cls");
			    system("color 40");
				printf("��ϲ��¶��ˣ�����\n");
               if(g==1)
                {
                    printf("�ϰ�����ͷ��ͷ������ң��������%d��Ӵ��\n",g);
                }else if(1<g<31)
                {
                    printf("���񻹴�Ӵ���������%d��Ӵ��\n",g);
                }else if(30<g<71)
                {
                    printf("������Ӵ�����������%d��Ӵ��\n",g);
                }else
                {
                    printf("������������������ɣ���\n",g);
                }
                g=0;
				windows_1();
				scanf("%d", &z);
				if (z == 1)
				{
				    system("color B0");
				    srand((unsigned)time(NULL));
				    random_num= rand() % 1000+1;

				}
				else
				{
				    printf("�س�������˳�");
					break;
				}
				continue;
			}
		}
}
void function_case2_2()
{

        printf("������һ����д��ĸ���в�ȡ��\n");
        for(g=1;;g++)
        {
        printf("��%d�����룡",g);
        getchar();
        scanf("%c",&playerinput2);

        if(random_num>playerinput2)
        {
            printf("����Ӵ���������������ĸ�ĺ���Ӵ�������룺\n");
        }else if(random_num<playerinput2)
        {
            printf("����Ӵ���������������ĸ��ǰ��Ӵ�������룺\n");
        }else
        {
            system("cls");
            system("color 40");
            printf("��ϲ��¶��ˣ�����\n");
            if(g==1)
                {
                    printf("���ѳ�Խ����֮�⣡�������%d��Ӵ��\n",g);
                }else if(1<g<11)
                {
                    printf("���񻹴�Ӵ���������%d��Ӵ��\n",g);
                }else if(10<g<30)
                {
                    printf("������Ӵ�����������%d��Ӵ��\n",g);
                }else
                {
                    printf("������������������ɣ���\n",g);
                }
            g=0;
            windows_1();
            scanf("%d", &z);
            if (z == 2)
            {
                printf("�س�������˳�");
                break;
            }
            else
            {
                system("color B0");
                srand((unsigned)time(NULL));
                random_num= rand() % 25+65;
            }
        }
    }
}

void windows_1()
{
    printf("**************************************************************\n");
    printf("*************    ��      ��      ��      ��    ***************\n");
    printf("*****                                                    *****\n");
    printf("*****                                                    *****\n");
    printf("*****            1.��                                    *****\n");
    printf("*****                                                    *****\n");
    printf("*****            2.��                                    *****\n");
    printf("*****                                                    *****\n");
    printf("**************************************************************\n");
    printf("**************************************************************\n");

}
