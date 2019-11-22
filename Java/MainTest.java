package com.jdbc.test;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import com.jdbc.util.JDBCUtil;


public class MainTest {

	public static void main(String[] args) {
		
		Connection conn = null;
		Statement st = null;
		ResultSet rs = null;
		//查询
	
		try {
			//1.获取连接对象
			conn = JDBCUtil.getConn();
			
			//2.根据链接对象，得到statement
			st = conn.createStatement();
			
			//3.执行sql语句,返回resultset
			String sql = "select * from users";
			rs = st.executeQuery(sql);
			
			//.4.遍历结果集
			while (rs.next()) {
				String id = rs.getString("id");
				int lv = rs.getInt("lv");
				String site = rs.getString("site");
				int answer = rs.getInt("answerNum");
				int topAnswerNum = rs.getInt("topAnswerNum");
				double accept = rs.getDouble("accept");
				int attention = rs.getInt("attention");
				int fans = rs.getInt("fans");
				int bee = rs.getInt("bee");
				System.out.println(id + "\t\t" + lv + "\t" + site + "\t" + answer + "\t" + topAnswerNum + "\t" + accept + "\t"
										+ attention + "\t" + fans + "\t" + bee);
				
				
			}
			
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			JDBCUtil.release(conn, st, rs);
		}
		
		
		
	}

}
