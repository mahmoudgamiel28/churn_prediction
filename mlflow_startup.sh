#!/bin/bash
set -e

echo "Starting MLflow setup..."

# Create directories
mkdir -p /mlflow/artifacts
chmod 777 /mlflow
chmod 777 /mlflow/artifacts

# Initialize database using Python (more reliable than sqlite3 command)
echo "Initializing database with Python..."
python3 -c "
import sqlite3
import os

db_path = '/mlflow/mlflow.db'
print(f'Creating database at {db_path}')

# Create database file
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Test the connection
cursor.execute('CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)')
cursor.execute('DROP TABLE IF EXISTS test_table')
conn.commit()
conn.close()

# Set permissions
os.chmod(db_path, 0o666)
print('Database initialization successful!')
"

# Start MLflow server
echo "Starting MLflow server..."
exec mlflow server \
    --backend-store-uri "sqlite:///mlflow/mlflow.db" \
    --default-artifact-root "/mlflow/artifacts" \
    --host "0.0.0.0" \
    --port "5050"