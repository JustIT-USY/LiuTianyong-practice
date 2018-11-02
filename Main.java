package 点名器;

import java.util.Random;
import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int last = -1;
		int i = 0;
		String[] students = {"龚家浩","张永桥","陈  磊"};
		int[] studentID = {2017001,2017002,2017003};
		System.out.println("----------随机点名器----------\n");
		printStudentName(students, studentID);
//		String name = randomStudentName(students, last);
//		for (i = 0; i < studentID.length; i++) {
//			if (students[i].equalsIgnoreCase(name)) {
//				last = i;
//				break;
//			}
//		}
//		System.out.println("\n被点到的人是：" + students[i] + "\t学号是："+studentID[i]);
//		System.out.println("是否继续点名:");
//		String p = new Scanner(System.in).next();
		String p = "是";
		while ("是".equalsIgnoreCase(p)) {
			String name = randomStudentName(students, last);
			for (i = 0; i < studentID.length; i++) {
				if (students[i].equalsIgnoreCase(name)) {
					last = i;
					break;
				}
			}
			System.out.println("\n被点到的人是：" + students[i] + "\t学号是："+studentID[i]);
			System.out.print("是否继续点名:");
			
			p = new Scanner(System.in).next();
		}
		System.out.println("退 出 程 序!");
	}
	
	public static void printStudentName(String[] students,int[] studentID) {
		for (int i = 0; i < studentID.length; i++) {
			System.out.println("姓名；" + students[i] + "\t学号："+studentID[i]);
		}
	}
	/*
	 * 随机点名
	 */
	public static String randomStudentName(String[] students,int last) {
		int index = new Random().nextInt(students.length);
		while (index == last) {
			index = new Random().nextInt(students.length);
		}
		return students[index]; 			//返回学生索引
	}

}

