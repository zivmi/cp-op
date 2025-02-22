import logging
import os
import sys
import sqlite3
import pandas as pd

# Set up logging
logging_file = '/logs/data_processing_pt1.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logging_file, encoding='utf-8', level=logging.INFO)

# Specify the paths to the database and the SQL script
database_path = "option_prices_raw.db"  
sql_script_path = "preprocessing_script.sql"  

conn = sqlite3.connect(database_path)

# Read and execute the SQL script
try:
    with open(sql_script_path, "r") as sql_file:
        sql_script = sql_file.read()  # Read the SQL script into a string
    
    # Execute the SQL script
    with conn:
        conn.executescript(sql_script)
        print("SQL script executed successfully.")
finally:
    conn.close()  # Ensure the connection is closed