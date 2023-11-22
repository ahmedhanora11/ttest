import psycopg2

# Define the connection parameters
db_params = {
    "dbname": "AIattend",
    "user": "postgres",
    "password": "dede7dede",  # Add your database password if required
    "host": "localhost",  # Change to your database host if it's not running locally
    "port": "5432",  # Change to your database port if different
}

# Establish a connection to the database
conn = psycopg2.connect(**db_params)

# Create a cursor
cursor = conn.cursor()

# Get a list of all tables in the 'public' schema
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
""")
table_names = [row[0] for row in cursor.fetchall()]

# Fetch and display data for each table
for table_name in table_names:
    print(f"Data in Table: {table_name}")
    
    # Execute a SELECT query for the current table
    cursor.execute(f"SELECT * FROM {table_name}")
    
    # Fetch all rows in the table
    rows = cursor.fetchall()
    
    # Get the column names
    column_names = [desc[0] for desc in cursor.description]
    
    # Print column names
    print(" | ".join(column_names))
    
    # Print the data in the table
    for row in rows:
        print(" | ".join(str(value) for value in row))
    
    print("")

# Close the cursor and the connection
cursor.close()
conn.close()
