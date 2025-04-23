# src/utils/database.py
import os
import json
import sqlite3
from datetime import datetime

# Ensure database directory exists
os.makedirs('data/db', exist_ok=True)

# Database file path
DB_FILE = 'data/db/road_safety.db'

def setup_database():
    """Setup SQLite database schema"""
    print("Setting up database schema...")
    
    # Connect to SQLite database (will be created if it doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create accidents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            accident_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            latitude REAL,
            longitude REAL,
            severity INTEGER,
            casualties INTEGER DEFAULT 0,
            vehicles_involved INTEGER DEFAULT 0,
            weather_condition TEXT,
            road_condition TEXT,
            light_condition TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create road_segments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS road_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT UNIQUE NOT NULL,
            start_lon REAL,
            start_lat REAL,
            end_lon REAL,
            end_lat REAL,
            road_type TEXT,
            name TEXT,
            lanes INTEGER,
            speed_limit INTEGER,
            oneway INTEGER,
            surface TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indices for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_timestamp ON accidents(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_severity ON accidents(severity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_location ON accidents(latitude, longitude)")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database schema created successfully")

def load_sample_data():
    """Load sample data into SQLite database"""
    print("Loading sample data into database...")
    
    # Check if sample data exists
    if not os.path.exists("data/sample/accidents.json") or not os.path.exists("data/sample/road_network.json"):
        print("Error: Sample data files not found. Run data generator first.")
        return False
    
    # Connect to SQLite database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Load accidents data
    with open("data/sample/accidents.json", 'r') as f:
        accidents = json.load(f)
    
    print(f"Loading {len(accidents)} accident records...")
    
    # First, clear existing data (optional)
    cursor.execute("DELETE FROM accidents")
    
    # Insert accident data
    for accident in accidents:
        cursor.execute('''
            INSERT OR REPLACE INTO accidents (
                accident_id, timestamp, latitude, longitude,
                severity, casualties, vehicles_involved,
                weather_condition, road_condition, light_condition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            accident["accident_id"],
            accident["timestamp"],
            accident["latitude"],
            accident["longitude"],
            accident["severity"],
            accident["casualties"],
            accident["vehicles_involved"],
            accident["weather_condition"],
            accident["road_condition"],
            accident["light_condition"]
        ))
    
    # Load road network data
    with open("data/sample/road_network.json", 'r') as f:
        road_segments = json.load(f)
    
    print(f"Loading {len(road_segments)} road segments...")
    
    # First, clear existing data (optional)
    cursor.execute("DELETE FROM road_segments")
    
    # Insert road segment data
    for segment in road_segments:
        cursor.execute('''
            INSERT OR REPLACE INTO road_segments (
                segment_id, start_lon, start_lat, end_lon, end_lat,
                road_type, name, lanes, speed_limit, oneway, surface
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            segment["segment_id"],
            segment["start_lon"],
            segment["start_lat"],
            segment["end_lon"],
            segment["end_lat"],
            segment["road_type"],
            segment["name"],
            segment["lanes"],
            segment["speed_limit"],
            segment["oneway"],
            segment["surface"]
        ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Sample data loaded successfully")
    return True

def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

if __name__ == "__main__":
    print("\n=== DATABASE SETUP ===\n")
    
    # Setup database
    setup_database()
    
    # Load sample data
    success = load_sample_data()
    
    if success:
        print("\nDatabase setup complete!")
    else:
        print("\nDatabase setup incomplete. Please generate sample data first.")
