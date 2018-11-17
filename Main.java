package sqlDemo;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

import com.mysql.jdbc.PreparedStatement;

public class Main {
	public static void main(String[] args) {


		Connection conn = null;
		PreparedStatement ps = null;	
		
		try {
		    conn =
		       DriverManager.getConnection("jdbc:mysql://localhost/mydb?" +
		                                   "user=root&password=");

		    // Do something with the Connection

		} catch (SQLException ex) {
		    // handle any errors
		    System.out.println("SQLException: " + ex.getMessage());
		    System.out.println("SQLState: " + ex.getSQLState());
		    System.out.println("VendorError: " + ex.getErrorCode());
		}
		System.out.println(ps);
		System.out.println(conn);

	}
	
}
