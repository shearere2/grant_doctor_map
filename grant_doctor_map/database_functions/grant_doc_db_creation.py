import sqlite3


query = '''
CREATE TABLE IF NOT EXISTS npi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lastname VARCHAR(100) NOT NULL,
    forename VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''

conn = sqlite3.connect('data/grant_npi.db')
cursor = conn.cursor()
cursor.execute(query)
cursor.close()