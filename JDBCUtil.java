package com.jdbc.uitl;

import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;


public class JDBCUtil {
	
	static String driverClass = null;
	static String url = null;
	static String name = null;
	static String password = null;
	
	static {
		
		try {
			//创建一个属性配置对象
			Properties properties = new Properties();
			//InputStream is =new FileInputStream("jdbc.properties");
			InputStream is = JDBCUtil.class.getClassLoader().getResourceAsStream("jdbc.properties");
			//导入输入流
			properties.load(is);
			//读取属性
			driverClass = properties.getProperty("driverClass");
			url = properties.getProperty("url");
			name = properties.getProperty("name");
			password = properties.getProperty("password");
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	/*
	 * 获取连接对象
	 */
	public static Connection getConn() {
		
		Connection conn = null;
		try {
			Class.forName(driverClass).newInstance();
			conn = DriverManager.getConnection(url,name,password);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//写法二:注册两次,不赞同该写法,Driver内有静态代码块，导致注册两次
		//DriverManager.registerDriver(new com.mysql.jdbc.Driver());
		/*
		 * 2.建立连接
		 * 参数一：协议 + 访问数据库
		 * 参数二：用户名
		 * 参数三：密码
		 */
		//conn = DriverManager.getConnection("jdbc:mysql://localhost/mydb?user=root&password=123456");
		return conn;
		
	}
	
	/*
	 * @释放资源
	 * 
	 */
	public static void release(Connection conn, Statement st, ResultSet rs) {
		closeRs(rs);
		closeSt(st);
		closeConn(conn);
	}
	
	private static void closeRs(ResultSet rs) {
		try {
			if (rs != null) {
				rs.close();
			}
		} catch (Exception e2) {
			e2.printStackTrace();	
		}finally {
			rs = null;
		}
	}
	
	private static void closeSt(Statement st) {
		try {
			if (st != null) {
				st.close();
			}
		} catch (Exception e2) {
			e2.printStackTrace();	
		}finally {
			st = null;
		}
	}
	
	private static void closeConn(Connection conn) {
		try {
			if (conn != null) {
				conn.close();
			}
		} catch (Exception e2) {
			e2.printStackTrace();	
		}finally {
			conn = null;
		}
	}
}
