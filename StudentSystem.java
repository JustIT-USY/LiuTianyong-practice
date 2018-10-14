package 学生成绩管理系统;

import java.util.Scanner;

public class StudentSystem {
	
	static Student [] stu = new Student[100];
	static Scanner sc = new java.util.Scanner(System.in);

	public static void main(String[] args) {
		int studentSum = 0;
		while (true) {
			System.out.println("\t\t\t欢迎来到学生成绩管理系统");
			System.out.println("\t\t1.信息添加");
			System.out.println("\t\t2.信息查看");
			System.out.println("\t\t3.信息修改");
			System.out.println("\t\t4.信息删除");
			System.out.println("\t\t5.信息按成绩从低到高排序");  
			System.out.println("\t\t6.信息查询");
			System.out.println("\t\t7.退出");
			System.out.print("\t\t请输入你要进行的操作:");
			int choice = sc.nextInt();
			if(choice == 7) {
				break;
			}
			else {
				switch (choice) {
				case 1://增加学生成绩
					System.out.print("请问你需要增加几个学生的成绩；");
					int studentNum = sc.nextInt();
					addStudent(studentNum, studentSum);
					break;
				case 2://查看学生成绩
					showStudent();
					break;
				case 3://信息修改
					amendStudent();
					break;
				case 4://删除
					deleteStudent();
					break;
				case 5:
					sortStudent();
					break;
				case 6:
					inquireStudent();
					break;
				case 7:
					System.out.println("欢迎下次使用!");
					System.exit(0);
					break;
				default:
					System.out.println("输入错误");
					break;
				}
			}
		}
	}
	
	static void addStudent(int studentNum, int studentSum){//添加学生信息
		for (int i = studentSum; i < studentNum + studentSum; i++) {
			stu[i] = new Student();
			System.err.print("请输入学生学号:");
			stu[i].setStudentId(sc.nextLong());
			System.out.print("请输入学生姓名:");
			stu[i].setName(sc.next());
			System.out.print("请输入学生年龄");
			stu[i].setAge(sc.nextInt());
			System.out.print("请输入学生性别");
			stu[i].setSex(sc.next());
			System.out.print("请输入学生语文成绩:");
			stu[i].setChineseScore(sc.nextInt());
			System.out.print("请输入学生数学成绩:");
			stu[i].setMathScore(sc.nextInt());
			System.out.print("请输入学生英语成绩:");
			stu[i].setEnglishScore(sc.nextInt());
		}
		System.err.println("录入完毕");
	}
	static void showStudent() {
		System.out.println("学号\t姓名\t年龄\t性别\t语文\t数学\t英语");
		for (int i = 0; i < stu.length; i++) {
			System.out.println(stu[i].getStudentId()+ "\t"+stu[i].getName()+"\t"+stu[i].getAge()+"\t"+stu[i].getSex()+"\t"+stu[i].getChineseScore()+"\t"+stu[i].getMathScore()+"\t"+stu[i].getEnglishScore());
		}
	}
	static void amendStudent() {
		System.out.print("请输入你需要修改的学生姓名:");
		String name = sc.next();
		for (int i = 0; i < stu.length; i++) {
			if (stu[i].getName().equalsIgnoreCase(name)) {
				System.out.println("请问重新输入该人信息:");
				addStudent(i, 1);
			}
		}
	}
	static void deleteStudent() {
		System.err.print("你需要删除学生的姓名是:");
		String name = sc.next();
		for (int i = 0; i < stu.length; i++) {
			if (stu[i].getName().equalsIgnoreCase(name)) {
				for (int j = i; j < stu.length; j++) {
					stu[i] = stu[i+1];
				}
			}
		}
	}
	static void sortStudent() {
		for (int i = 0; i < stu.length; i++) {
			for (int j = i; j < stu.length; j++) {
				if (stu[j].getChineseScore() < stu[j+1].getChineseScore()) {
					int temp = stu[j].getChineseScore();
					stu[j].setChineseScore(stu[j+1].getChineseScore());
					stu[j+1].setChineseScore(temp);
				}
			}
		}
	}
	static void inquireStudent() {
		System.out.println("请输入你要查询人的姓名:");
		String name = sc.next();
		for (int i = 0; i < stu.length; i++) {
			if (stu[i].getName().equalsIgnoreCase(name)) {
				System.out.println(stu[i].getStudentId()+ "\t"+stu[i].getName()+"\t"+stu[i].getAge()+"\t"+stu[i].getSex()+"\t"+stu[i].getChineseScore()+"\t"+stu[i].getMathScore()+"\t"+stu[i].getEnglishScore());
			}
		}
	}
	
}
