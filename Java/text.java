package com.jdbc.test;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.junit.Test;

import com.jdbc.util.JDBCUtil;

/**
 * 使用junit执行单元测试
 */
public class text {
	@Test
	public void testQuery() {

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
	@Test
	public void insert() {
		Connection conn = null;
		Statement st = null;
		
		//查询
	
		try {
			//1.获取连接对象
			conn = JDBCUtil.getConn();
			
			//2.根据链接对象，得到statement
			st = conn.createStatement();
			
			//3.执行sql语句,返回resultset
			String sql = "insert into users values('ggg' ,20, 'gg', 10, 2, 2, 55, 66, 10)";
			int result = st.executeUpdate(sql);
			System.out.println(result);
			//大于0 即为1 操作成功 
			if(result > 0) {
				System.out.println("添加成功");
			}else {
				System.out.println("添加失败");
			}
			
			
		} catch (SQLException e) {
			
			e.printStackTrace();
		} finally {
			
			JDBCUtil.release(conn, st, null);
		}
	}
	@Test
	public void update() {
		Connection conn = null;
		Statement st = null;
		
		//查询
	
		try {
			//1.获取连接对象
			conn = JDBCUtil.getConn();
			
			//2.根据链接对象，得到statement
			st = conn.createStatement();
			
			//3.执行sql语句,返回resultset
			String sql = "update users set lv = 26 where id='静静'";
			int result = st.executeUpdate(sql);
			System.out.println(result);
			//大于0 即为1 操作成功 
			if(result > 0) {
				System.out.println("更新成功");
			}else {
				System.out.println("更新失败");
			}
			
		} catch (SQLException e) {
			
			e.printStackTrace();
		} finally {
			
			JDBCUtil.release(conn, st, null);
		}
	}
	@Test
	public void detele() {
		Connection conn = null;
		Statement st = null;
		
		//查询
	
		try {
			//1.获取连接对象
			conn = JDBCUtil.getConn();
			
			//2.根据链接对象，得到statement
			st = conn.createStatement();
			
			//3.执行sql语句,返回resultset
			String sql = "detele from users where id='静静'";
			int result = st.executeUpdate(sql);
			System.out.println(result);
			//大于0 即为1 操作成功 
			if(result > 0) {
				System.out.println("删除成功");
			}else {
				System.out.println("删除失败");
			}
			
		} catch (SQLException e) {
			
			e.printStackTrace();
		} finally {
			
			JDBCUtil.release(conn, st, null);
		}
	}

}
