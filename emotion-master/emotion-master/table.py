import psycopg2

# Define the database connection parameters
connection_params = {
    "host": "localhost",
    "user": "postgres",
    "password": "dede7dede",
    "port": "5432"
}

database_name = "AIattend"

# Data to be inserted
val = [
  ('C001', 'Imran Sarwar', 39, 'CEO'),
  ('H002', 'Aliaa Balqis', 42, 'Human Resource'),
  ('F003', 'Elina Rahim', 35, 'Finance Cum Project Exec'),
  ('DS04', 'Anati Bidin', 24, 'Data Scientist'),
  ('DS05', 'Hui Shan', 23, 'Data Scientist'),
  ('AI06', 'Mardhiah Nasri', 24, 'AI Engineer'),
  ('UI07', 'Izlin Syamira', 22, 'UI/UX Designer'),
  ('UI08', 'Aisya Azhar', 22, 'UI/UX Engineer'),
  ('T009', 'Yasmira Husna', 22, 'Technology Executive'),
  ('B010', 'Ahmedalla Hani', 22, 'Backend Engineer'),
  ('A011', 'Rafiq Hisham', 22, 'AIoT'),
  ('M012', 'Nur Halifah', 22, 'Marketing')
]

try:
    # Establish a connection to the PostgreSQL server (connects to the 'postgres' database by default)
    conn = psycopg2.connect(**connection_params)

    # Create a cursor object
    cursor = conn.cursor()

    # Check if the database exists, and create it if it doesn't
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (database_name,))
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {database_name};")
        conn.commit()

    # Connect to the specific database
    conn.close()
    connection_params["database"] = database_name
    conn = psycopg2.connect(**connection_params)

    # Create a cursor object for the specific database
    cursor = conn.cursor()

    # Check if the table exists, and create it if it doesn't
    cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'employee');")
    if not cursor.fetchone()[0]:
        cursor.execute("CREATE TABLE employee (employeeID VARCHAR(5), name VARCHAR(255), age INT, position VARCHAR(255));")
        conn.commit()

    # Insert data into the table
    sql = "INSERT INTO employee (employeeID, name, age, position) VALUES (%s, %s, %s, %s)"
    cursor.executemany(sql, val)
    conn.commit()

    print(cursor.rowcount, "rows were inserted.")

except psycopg2.Error as e:
    print(f"Error: {e}")

finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()
