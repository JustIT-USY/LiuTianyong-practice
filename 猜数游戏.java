package 猜数游戏;

import java.util.Scanner;
import java.util.Random;
import java.lang.String;

public class Main {
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(System.in);
		int i = menuShowOne();
		judge(i);
	}
	public static int menuShowOne() {
		System.out.println("\t\t欢迎来到猜数游戏");
		System.out.println("-----------------------------------------------");
		System.out.println("|*********************************************|");
		System.out.println("|***************1.初级猜数游戏****************|");
		System.out.println("|***************2.中级猜数游戏****************|");
		System.out.println("|***************3.高级猜数游戏****************|");
		System.out.println("|*********************************************|");
		System.out.println("-----------------------------------------------");
		System.out.print("-->请选择游戏难度:");
		Scanner sc = new Scanner(System.in);
		int i = sc.nextInt();
		return i;
	}
	public static void game(int randomNum, int j) {
		Random rd = new Random();
		int SystemNum =  (int) (rd.nextDouble() * randomNum);
		System.out.print("请输入你猜的数字:");
		for(int i = 0;i < j * 5;i++) {
			Scanner sc = new Scanner(System.in);
			int playerNum = sc.nextInt();
			if(playerNum == SystemNum) {
				System.out.println("恭喜你猜对了!");
				System.out.print("是否再来一局:");
				String inputString = sc.next();
				if (inputString.equalsIgnoreCase("是")) {
					int j1 = menuShowOne();
					judge(j1);
				}else {
					System.out.println("退出游戏");
				}
				
			}else if (playerNum > SystemNum) {
				System.out.println("你猜大了!");
			}else {
				System.out.println("你猜小了!");
			}
			System.out.print("请重新输入:");
		}
		System.out.println("退出游戏");
		System.exit(0);
	}
	public static void judge(int k) {
		if(k == 1) {
			game(10, k);
		}else if (k == 2) {
			game(100, k);
		}else {
			game(1000, k);
		}
	}
}
