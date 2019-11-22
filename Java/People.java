package 模拟超市管理系统;

/**
 * @author 刘天勇
 *
 */
public class People {
	private String name;

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}
	
	public void shopping(String cd, String spName) {
		System.out.println(this.name +"到" + spName + "超市购买" + cd);
	}
}
