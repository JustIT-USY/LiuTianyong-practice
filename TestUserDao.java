package com.jdbc.test;

import org.junit.Test;

import com.jdbc.dao.UsersDao;
import com.jdbc.dao.impl.UserDaoImpl;

public class TestUserDao {
	
	@Test
	public void testFindAll() {
		UsersDao daoImpl = new UserDaoImpl();
		daoImpl.findAll();
	}
	@Test
	public void testLogin() {
		UsersDao dao = new UserDaoImpl();
		String userName = "admin";
		String passWord = "10086";
		dao.login(userName, passWord);
	}
}
