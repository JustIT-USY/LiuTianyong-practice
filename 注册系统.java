package 注册系统;

import java.lang.reflect.WildcardType;
import java.util.Scanner;

public class Mian {
	
	static String user = "abcdefg";
	static String password = "aaaaaa";
	static String email = "123456@qq.com";
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("************************欢迎登陆***********************");
		System.out.println("*                  1.登       陆                      *");
		System.out.println("*******************************************************");
		System.out.println("*                  2.注       册                      *");
		System.out.println("*******************************************************");
		System.out.println("*                  3.忘 记 密 码                      *");
		System.out.println("*******************************************************");
		System.out.print("请输入你要进行的操作:");
		Scanner sc = new Scanner(System.in);
		while (true) {
			int choice = sc.nextInt();
			switch (choice) {
			case 1:
				System.out.println("进行登陆");
				login();
				break;
			case 2:
				System.out.println("进行注册");
				register();
				break;
			case 3:
				retrievePassword();
				break;
			default:
				System.out.print("输入有误，请重新输入:");
				break;
			}
			
		}
		
	}
	public static void login() {
		for (int i = 0; i <=  3; i++) {
			if (i == 3) {
				System.out.println("登陆次数上限,请明日再来!");
				System.exit(0);
			}
			System.out.print("请输入你的账户:");
			Scanner sc = new Scanner(System.in);
			String inputUser = sc.next();
			System.out.print("请输入的密码:");
			String inputPassword = sc.next();
			if (inputUser.equalsIgnoreCase(user) & inputPassword.equalsIgnoreCase(password)) {
				System.out.println("登陆成功!");
			}else {
				System.out.println("登陆失败,请重新登陆!");
			}
		}
		
	}
	public static void register() {
		Scanner sc = new Scanner(System.in);
		while (true) {
			System.out.print("请输入你的注册邮箱:");
			email = sc.next();
			System.out.print("请输入你的账户:");
			user = sc.next();
			System.out.print("请输入你的密码:");
			password = sc.next();
			System.out.print("请确认你的密码:");
			String newPassword = sc.next();
			if (email.indexOf('@') == -1) {
				System.out.println("你输入的邮箱不合法,请重新注册。");
			}else if (user.length() >= 6 & password.length() >= 6 & newPassword.equalsIgnoreCase(password)) {
				System.out.println("注册成功");
				break;
			}
			
		}
		
	}
	public static void retrievePassword() {
		System.out.print("请输入你的注册邮箱:");
		Scanner sc = new Scanner(System.in);
		String inputEmail = sc.next();
		if (inputEmail.equalsIgnoreCase(email)) {
			System.out.println("你的账户为:"+user);
			System.out.print("请输入新密码:");
			String newPassword = sc.next();
			System.out.print("请确定密码:");
			password = sc.next();
			if (newPassword.equalsIgnoreCase(password)) {
				System.out.println("密码重置成功，请牢记密码!");
			}
		}
		
	}

}
