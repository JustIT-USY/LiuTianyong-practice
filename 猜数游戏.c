#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
/*对进阶版猜数游戏结构进行优化
增加登陆注册等功能
*/
/*************************************************
Function:猜数游戏加强版
Description:可以猜数字，猜字母
Output:
Return:
Others:
auther:刘天勇
*************************************************/
int random_num;//随机赋值变量
int g,z;//次数评价
int playerinput;
char playerinput2;
void function_case1_1();
void function_case1_2();
void function_case1_3();
void function_case2_2();
void windows_1();
int main()
{   system("color A1");
    //随机赋值变量
    int user,password;
    int input_randomnum;//玩家输入的验证码
    int input_user,input_PW;
    int i=0;//循环变量
    int x,y;//菜单变量
    printf("欢迎来到猜数游戏！\n");
    printf("请注册游戏账号\n");
    do{
        srand((unsigned)time(NULL));
        random_num=(rand()*rand())%10000+1;
        printf("请输入你要注册的账户（10位纯数字）：\n");
        scanf("%d",&user);
        printf("请输入你的密码（10位纯数字）：\n");
        scanf("%d",&password);
        printf("请输入验证码%d\t验证码：",random_num);
        scanf("%d",&input_randomnum);
        if(input_randomnum!=random_num)
        {
            printf("验证码输入错误，请重新注册！\n");
        }
    }while(input_randomnum!=random_num);
    system("cls");
    printf("你的账户是：%d\n你的密码是：%d\n请牢记！！！",user,password);
    Sleep(700);
    system("cls");
    system("color A0");
    printf("请登录！\n");
    do{
        srand((unsigned)time(NULL));
        random_num=(rand()*rand())%10000+1;
        if(i==3)
        {
            printf("你的次数已经用完，请核对信息后再来！");
            system("exit");
        }
        i++;
        if(i>1)
        {
            printf("账户或密码不正确，请重新输入!\n");
        }

        printf("账号： ");
        scanf("%d",&input_user);
        printf("密码： ");
        scanf("%d",&input_PW);
        printf("你的验证码是%d\t",random_num);
        scanf("%d",&input_randomnum);

    }while((input_user!=user||input_PW!=password||input_randomnum!=random_num)&&i<4);

    printf("登陆成功！");
    Sleep(700);
    system("cls");
    printf("**************************************************************\n");
	printf("************* 欢  迎  进  入  猜  字  游  戏 *****************\n");
	printf("*****                                                    *****\n");
	printf("*****                                                    *****\n");
	printf("*****            1.猜    字    游    戏                  *****\n");
	printf("*****                                                    *****\n");
	printf("*****            2.猜  字  母   游   戏                  *****\n");
	printf("*****                                                    *****\n");
	printf("*****            3.退    出    游    戏                  *****\n");
	printf("*****                                                    *****\n");
	printf("*****                                                    *****\n");
	printf("**************************************************************\n");
	printf("**************************************************************\n");
	scanf("%d", &x);
	system("cls");//系统清屏
	switch (x)
	{
    case 1:
        system("color B0");
		printf("**************************************************************\n");
		printf("************* 请   选   择   游   戏   难   度 ***************\n");
		printf("*****                                                    *****\n");
		printf("*****                                                    *****\n");
		printf("*****            1.初   级   难   度                     *****\n");
		printf("*****                                                    *****\n");
		printf("*****            2.中   级   难   度                     *****\n");
		printf("*****                                                    *****\n");
		printf("*****            3.高   级   难   度                     *****\n");
		printf("*****                                                    *****\n");
		printf("*****                                                    *****\n");
		printf("**************************************************************\n");
		printf("**************************************************************\n");
		scanf("%d", &y);
		system("cls");
        switch (y)
        {
        case 1://初级难度
            srand((unsigned)time(NULL));
            random_num=rand()%10+1;
            function_case1_1();
            break;
        case 2://中级难度
            srand((unsigned)time(NULL));
            random_num=rand()%100+1;
            function_case1_2();
            break;
        case 3://高级难度
            srand((unsigned)time(NULL));
            random_num=rand()%1000+1;
            function_case1_3();
            break;
        default:
            break;
        }
        break;
    case 2://猜字母游戏
        system("color 20");
        srand((unsigned)time(NULL));
        random_num=rand()%25+65;//随机数65~90
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
		    printf("第%d次输入！",g);
			printf("请输入1~10的数字来猜取：\n");
			scanf("%d", &playerinput);
			if (playerinput > random_num)
			{
				printf("你输入的数字大了哟！\t");
			}
			else if (playerinput < random_num)
			{
				printf("你输入的数字小了哟！\t");
			}
			else
			{
			    system("cls");
			    system("color 40");
				printf("恭喜你猜对了！！！\n");
				if(g==1)
                {
                    printf("大佬你已看破红尘，此难度以及不适合你了！你仅用了%d次哟！\n",g);
                }else if(1<g<4)
                {
                    printf("大神还错哟！你仅用了%d次哟！\n",g);
                }else if(3<g<6)
                {
                    printf("还可以哟！！你仅用了%d次哟！\n",g);
                }else
                {
                    printf("请喝完六个核桃再来吧！！\n");
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
				    printf("回车后程序退出");
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
		    printf("第%d次输入！",g);
			printf("请输入1~10的数字来猜取：\n");
			scanf("%d", &playerinput);
			if (playerinput > random_num)
			{
				printf("你输入的数字大了哟！\t");
			}
			else if (playerinput < random_num)
			{
				printf("你输入的数字小了哟！\t");
			}
			else
			{
			    system("cls");
			    system("color 40");
				printf("恭喜你猜对了！！！\n");
                if(g==1)
                {
                    printf("你已超越六界之外！你仅用了%d次哟！\n",g);
                }else if(1<g<11)
                {
                    printf("大神还错哟！你仅用了%d次哟！\n",g);
                }else if(10<g<30)
                {
                    printf("还可以哟！！你仅用了%d次哟！\n",g);
                }else
                {
                    printf("请喝完六个核桃再来吧！！\n",g);
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
				    printf("回车后程序退出");
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
		    printf("第%d次输入！",g);
			printf("请输入1~10的数字来猜取：\n");
			scanf("%d", &playerinput);
			if (playerinput > random_num)
			{
				printf("你输入的数字大了哟！\t");
			}
			else if (playerinput < random_num)
			{
				printf("你输入的数字小了哟！\t");
			}
			else
			{
			    system("cls");
			    system("color 40");
				printf("恭喜你猜对了！！！\n");
               if(g==1)
                {
                    printf("老板请手头抱头交出外挂！你仅用了%d次哟！\n",g);
                }else if(1<g<31)
                {
                    printf("大神还错哟！你仅用了%d次哟！\n",g);
                }else if(30<g<71)
                {
                    printf("还可以哟！！你仅用了%d次哟！\n",g);
                }else
                {
                    printf("请喝完六个核桃再来吧！！\n",g);
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
				    printf("回车后程序退出");
					break;
				}
				continue;
			}
		}
}
void function_case2_2()
{

        printf("请输入一个大写字母进行猜取：\n");
        for(g=1;;g++)
        {
        printf("第%d次输入！",g);
        getchar();
        scanf("%c",&playerinput2);

        if(random_num>playerinput2)
        {
            printf("不对哟！它在你输入的字母的后面哟！请输入：\n");
        }else if(random_num<playerinput2)
        {
            printf("不对哟！它在你输入的字母的前面哟！请输入：\n");
        }else
        {
            system("cls");
            system("color 40");
            printf("恭喜你猜对了！！！\n");
            if(g==1)
                {
                    printf("你已超越六界之外！你仅用了%d次哟！\n",g);
                }else if(1<g<11)
                {
                    printf("大神还错哟！你仅用了%d次哟！\n",g);
                }else if(10<g<30)
                {
                    printf("还可以哟！！你仅用了%d次哟！\n",g);
                }else
                {
                    printf("请喝完六个核桃再来吧！！\n",g);
                }
            g=0;
            windows_1();
            scanf("%d", &z);
            if (z == 2)
            {
                printf("回车后程序退出");
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
    printf("*************    是      否      继      续    ***************\n");
    printf("*****                                                    *****\n");
    printf("*****                                                    *****\n");
    printf("*****            1.是                                    *****\n");
    printf("*****                                                    *****\n");
    printf("*****            2.否                                    *****\n");
    printf("*****                                                    *****\n");
    printf("**************************************************************\n");
    printf("**************************************************************\n");

}
