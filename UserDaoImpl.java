package com.jdbc.dao.impl;

import java.awt.desktop.AboutHandler;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import com.jdbc.dao.UsersDao;
import com.jdbc.uitl.JDBCUtil;

public class UserDaoImpl implements UsersDao {

	public void findAll() {
		Connection conn = null;
		Statement st = null;
		ResultSet rs = null;
		
		try {
			conn = JDBCUtil.getConn();
			st = conn.createStatement();
			String sql = "select * from t_user";
			rs = st.executeQuery(sql);
			
			while (rs.next()) {
				String userName = rs.getString("username");
				String passWord = rs.getString("password");
				System.out.println(userName + "==="+passWord);
				
			}
			
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			JDBCUtil.release(conn, st, rs);
		}
		
	}
	public void login(String userName, String passWord) {
		Connection conn = null;
		Statement st = null;
		ResultSet rs = null;
		
		try {
			conn = JDBCUtil.getConn();
			st = conn.createStatement();
			String sql = "select * from t_user where username='"+ userName + "' and password='"+passWord+"'";
			rs = st.executeQuery(sql);
			
			if (rs.next()) {
				System.out.println("登陆成功");
			}else {
				System.out.println("登陆失败");
			}
			
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			JDBCUtil.release(conn, st, rs);
		}
	}
}
