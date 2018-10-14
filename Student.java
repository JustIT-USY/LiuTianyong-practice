package 学生成绩管理系统;

import java.util.Scanner;

public class Student {
	private String name;
	private int age;
	private long studentId;
	private String sex;
	private int chineseScore;
	private int mathScore;
	private int englishScore;
	
	public Student() {
		// TODO Auto-generated constructor stub
		super();
	}
	
	public void setName(String name) {
		this.name = name;
	}
	public void setAge(int age) {
		this.age = age;
	}
	public void setStudentId(Long studentId) {
		this.studentId = studentId;
	}
	public void setSex(String sex) {
		this.sex = sex;
	}
	public void setChineseScore(int chineseScore) {
		this.chineseScore = chineseScore;
	}
	public void setMathScore(int mathScore) {
		this.mathScore = mathScore;
	}
	public void setEnglishScore(int english) {
		this.englishScore = english;
	}
	
	public String getName() {
		return this.name;
	}
	public int getAge() {
		return this.age;
	}
	public long getStudentId() {
		return this.studentId;
	}
	public String getSex() {
		return this.sex;
	}
	public int getChineseScore() {
		return this.chineseScore;
	}
	public int getMathScore() {
		return this.mathScore;
	}
	public int getEnglishScore() {
		return this.englishScore;
	}
}
