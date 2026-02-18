"""Inspect n8n SQLite database schema."""
import sqlite3
import os

db_path = r"E:\AgentNate\.n8n-instances\shared\.n8n\database.sqlite"

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cursor.fetchall()]

print("=" * 60)
print("N8N DATABASE SCHEMA")
print("=" * 60)

for table in tables:
    print(f"\nTABLE: {table}")
    print("-" * 40)

    # Get column info
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    for col in columns:
        col_id, name, dtype, notnull, default, pk = col
        pk_mark = " [PK]" if pk else ""
        print(f"  {name}: {dtype}{pk_mark}")

    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  Rows: {count}")

conn.close()
print("\n" + "=" * 60)
