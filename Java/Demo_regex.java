package 正则demo1;

public class Demo_regex {
	public static void main(String[] args) {
		System.out.println(checkQQ("0123456789"));
		
		String regex = "[1-9]\\d{4,14}";
		System.out.println("02553868".matches(regex));
	}
	/*
	 * 1.要求5-15位的数字
	 * 2.0不能开头
	 * 3.必须是数字
	 * 校验QQ
	 * 1.明确返回值类型为boolean
	 * 2.明确参数列表
	 */
	public static boolean checkQQ(String qq) {
		boolean flag = true;		//如果符合要求返回true，否则false
		if (qq.length() >= 5 && qq.length() <= 15) {
			if (!qq.startsWith("0")) {
				char[] arr = qq.toCharArray();		//将字符串转化为字符数组
				for (int i = 0; i < arr.length; i++) {
					char ch = arr[i];
					if (!(ch >= '0' && ch <= '9')) {
						flag = flag;			//不是数字
						break;
					}
				}
			}else {
				flag = false;		//以0开头不符合要求
			}
		}else {
			flag = false;
		}
		return flag;
	}
}
