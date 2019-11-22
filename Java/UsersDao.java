package com.jdbc.dao;


/**
 * 
 * @author Administrator
 *定义操作数据库的方法
 */
public interface UsersDao {
	/**
	 * 查询所有
	 */
	void findAll();
	void login(String userName, String passWord);
}
