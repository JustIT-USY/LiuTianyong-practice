package 模拟超市管理系统;

import java.util.Scanner;


public class test {
	public static void main(String[] args) {
		commodity [] cd = new commodity[3];
		Supermarket sp1 = new Supermarket();
		
		System.out.println("请输入超市名称:");
		String supName = new Scanner(System.in).next();
		sp1.setSupName(supName);
		
		System.out.println("请录入商品->");
		for (int i = 0; i < cd.length; i++) {
			cd[i] = new commodity();
			System.out.println("请输入商品名:");
			String cdName = new Scanner(System.in).next();
			cd[i].setCdName(cdName);
			System.out.println("请输入商品价格:");
			double cdPrice = new Scanner(System.in).nextDouble();
			cd[i].setCdPrice(cdPrice);
		}
		sp1.setCdNum(cd.length);
		
		System.out.println("请输入人的姓名；");
		String name = new Scanner(System.in).next();
		People p = new People();
		p.setName(name);
		System.out.println("你需要购买什么?");
		String sp = new Scanner(System.in).next();
		for (int i = 0; i < cd.length; i++) {
			if (sp.equalsIgnoreCase(cd[i].getCdName())) {
				p.shopping(cd[i].getCdName(), sp1.getSupName());
			}
		}
		
	}
}
