package 构造方法;

public class demo09 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		myDemo p = new myDemo();
		p.myprint();
	}

}

class myDemo {
	private String name;
	private int age;
	private String sex;
		
	public myDemo() {
		name = "0.0";
		age = 19;
		sex = "女";
		return;
	}
	
	public void myprint() {
		System.out.println("\n"+name+'\t'+age+'\t'+sex);
	}
	
}