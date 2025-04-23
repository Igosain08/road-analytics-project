import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import sqlite3
from datetime import datetime, timedelta

# Make sure necessary directories exist
os.makedirs('data/real', exist_ok=True)
os.makedirs('data/db', exist_ok=True)

# Database file path
DB_FILE = 'data/db/road_safety.db'

def import_osm_road_network(city, country, bbox=None, max_roads=2000):
    """
    Import road network from OpenStreetMap for a specific area
    
    Args:
        city: City name
        country: Country or state name
        bbox: Bounding box coordinates (optional)
        max_roads: Maximum number of road segments to import
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import LineString, Point
    except ImportError:
        print("Error: Required packages not installed")
        print("Install with: pip install osmnx geopandas shapely")
        return False, []
    
    print(f"Downloading road network for {city}, {country}...")
    
    # Get the road network for the city
    if bbox:
        G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type='drive')
    else:
        try:
            G = ox.graph_from_place(f"{city}, {country}", network_type='drive')
        except Exception as e:
            print(f"Error getting data for {city}, {country}: {e}")
            print("Trying with a smaller area...")
            try:
                # Try with a more specific query
                G = ox.graph_from_place(f"Downtown {city}, {country}", network_type='drive')
            except:
                print(f"Error: Could not get data for {city}, {country}")
                return False, []
    
    # Convert to GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G)
    
    # Process edges to create road segments
    road_segments = []
    
    # Limit the number of edges processed
    edge_count = 0
    
    for idx, edge in edges.iterrows():
        if edge_count >= max_roads:
            break
            
        u, v, key = idx
        
        # Get nodes for the edge
        from_node = nodes.loc[u]
        to_node = nodes.loc[v]
        
        # Get road properties
        road_type = edge.get('highway', 'unknown')
        if isinstance(road_type, list):
            road_type = road_type[0]
        
        # Map OSM road types to our categories
        road_types = {
            'motorway': 'primary',
            'trunk': 'primary',
            'primary': 'primary',
            'secondary': 'secondary',
            'tertiary': 'tertiary',
            'residential': 'residential',
            'living_street': 'residential',
            'unclassified': 'residential',
            'service': 'residential'
        }
        
        mapped_type = road_types.get(road_type, 'residential')
        
        # Get name
        name = edge.get('name', f"Road {u}-{v}")
        if isinstance(name, list):
            name = name[0]
        
        # Get speed limit if available
        speed_limit = edge.get('maxspeed', None)
        if speed_limit:
            if isinstance(speed_limit, list):
                speed_limit = speed_limit[0]
            # Try to extract numeric value
            try:
                speed_limit = int(''.join(filter(str.isdigit, str(speed_limit))))
            except:
                speed_limit = None
        
        # Set default speed limits based on road type if not available
        if not speed_limit:
            default_speeds = {
                'primary': 50,
                'secondary': 40,
                'tertiary': 30,
                'residential': 25
            }
            speed_limit = default_speeds.get(mapped_type, 30)
        
        # Get number of lanes if available
        lanes = edge.get('lanes', None)
        if lanes:
            if isinstance(lanes, list):
                lanes = lanes[0]
            # Try to extract numeric value
            try:
                lanes = int(''.join(filter(str.isdigit, str(lanes))))
            except:
                lanes = None
        
        # Set default lanes based on road type if not available
        if not lanes:
            default_lanes = {
                'primary': 3,
                'secondary': 2,
                'tertiary': 1,
                'residential': 1
            }
            lanes = default_lanes.get(mapped_type, 1)
        
        # Check if road is one-way
        oneway = 1 if edge.get('oneway', False) else 0
        
        # Get surface type if available
        surface = edge.get('surface', 'asphalt')
        if isinstance(surface, list):
            surface = surface[0]
        
        # Create a segment ID
        segment_id = f"OSM-{u}-{v}-{key}"
        
        # Create road segment
        segment = {
            'segment_id': segment_id,
            'start_lat': from_node['y'],
            'start_lon': from_node['x'],
            'end_lat': to_node['y'],
            'end_lon': to_node['x'],
            'road_type': mapped_type,
            'name': name,
            'lanes': lanes,
            'speed_limit': speed_limit,
            'oneway': oneway,
            'surface': surface
        }
        
        road_segments.append(segment)
        edge_count += 1
    
    # Insert into database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("DELETE FROM road_segments")
    
    # Insert road segment data
    for segment in road_segments:
        cursor.execute('''
            INSERT OR REPLACE INTO road_segments (
                segment_id, start_lat, start_lon, end_lat, end_lon,
                road_type, name, lanes, speed_limit, oneway, surface
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            segment["segment_id"],
            segment["start_lat"],
            segment["start_lon"],
            segment["end_lat"],
            segment["end_lon"],
            segment["road_type"],
            segment["name"],
            segment["lanes"],
            segment["speed_limit"],
            segment["oneway"],
            segment["surface"]
        ))
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Imported {len(road_segments)} road segments from OpenStreetMap (limited to {max_roads})")
    return True, road_segments

def generate_realistic_accidents(road_segments, num_accidents=2000):
    """Generate realistic synthetic accidents based on real road network"""
    print(f"Generating {num_accidents} realistic accidents...")
    
    try:
        from geopy.distance import geodesic
    except ImportError:
        print("Error: geopy package not installed")
        print("Install with: pip install geopy")
        return False
    
    # Risk factors by road type (probability multipliers)
    risk_factors = {
        'primary': 1.5,    # Higher risk on primary roads
        'secondary': 1.2,  # Medium risk on secondary roads
        'tertiary': 0.8,   # Lower risk on tertiary roads
        'residential': 0.5 # Lowest risk on residential roads
    }
    
    # Risk factors by number of lanes
    lane_factors = {
        1: 0.7,  # Single lane (lower risk)
        2: 1.0,  # Two lanes (reference)
        3: 1.3,  # Three lanes (higher risk)
        4: 1.5   # Four+ lanes (highest risk)
    }
    
    # Risk factors by speed limit
    def speed_factor(speed):
        if speed < 30:
            return 0.6
        elif speed < 40:
            return 0.9
        elif speed < 50:
            return 1.2
        else:
            return 1.5
    
    # Calculate total risk for each segment
    total_risk = 0
    segment_risks = []
    
    for segment in road_segments:
        # Calculate risk score for this segment
        r_type = risk_factors.get(segment['road_type'], 1.0)
        r_lanes = lane_factors.get(segment['lanes'], 1.0)
        r_speed = speed_factor(segment['speed_limit'])
        
        # Calculate segment length (approximate)
        start = (segment['start_lat'], segment['start_lon'])
        end = (segment['end_lat'], segment['end_lon'])
        length_km = geodesic(start, end).kilometers
        
        # Risk is proportional to segment length and other factors
        risk = length_km * r_type * r_lanes * r_speed
        
        segment_risks.append((segment, risk))
        total_risk += risk
    
    # Normalize risks to probabilities
    segment_probs = [(segment, risk / total_risk) for segment, risk in segment_risks]
    
    # Weather conditions with realistic probabilities
    weather_conditions = {
        'Clear': 0.6,
        'Cloudy': 0.15,
        'Rain': 0.1,
        'Snow': 0.05,
        'Fog': 0.05,
        'Unknown': 0.05
    }
    
    # Road conditions with realistic probabilities
    road_conditions = {
        'Dry': 0.7,
        'Wet': 0.15,
        'Snow': 0.05,
        'Ice': 0.05,
        'Slush': 0.03,
        'Unknown': 0.02
    }
    
    # Light conditions with realistic probabilities
    light_conditions = {
        'Daylight': 0.6,
        'Dark - lights on': 0.25,
        'Dark - no lights': 0.1,
        'Dawn/Dusk': 0.05
    }
    
    # Helper function for weighted random choice
    def weighted_choice(choices):
        total = sum(choices.values())
        r = np.random.uniform(0, total)
        upto = 0
        for choice, weight in choices.items():
            upto += weight
            if upto >= r:
                return choice
        return list(choices.keys())[0]  # Fallback
    
    # Generate accidents
    accidents = []
    
    # Get current date and time
    now = datetime.now()
    
    # Extract just the segments for probability selection
    segments = [s for s, _ in segment_probs]
    probs = [p for _, p in segment_probs]
    
    for i in range(num_accidents):
        # Select a road segment based on risk probability
        segment = np.random.choice(segments, p=probs)
        
        # Generate a position along the segment
        t = np.random.uniform(0, 1)
        lat = segment['start_lat'] + t * (segment['end_lat'] - segment['start_lat'])
        lon = segment['start_lon'] + t * (segment['end_lon'] - segment['start_lon'])
        
        # Generate timestamp within the last 2 years
        days_ago = np.random.randint(0, 365 * 2)
        hours_ago = np.random.randint(0, 24)
        minutes_ago = np.random.randint(0, 60)
        
        timestamp = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        # Rush hour is more likely
        hour = timestamp.hour
        if (5 <= hour <= 9) or (15 <= hour <= 19):  # Rush hours
            if np.random.random() < 0.3:  # 30% chance to skip this accident
                continue
        
        # Weather condition
        weather_condition = weighted_choice(weather_conditions)
        
        # Road condition (correlated with weather)
        if weather_condition == 'Rain':
            road_probs = road_conditions.copy()
            road_probs['Wet'] = 0.8  # Much higher chance of wet roads if raining
            road_condition = weighted_choice(road_probs)
        elif weather_condition == 'Snow':
            road_probs = road_conditions.copy()
            road_probs['Snow'] = 0.7  # Much higher chance of snowy roads if snowing
            road_probs['Ice'] = 0.2   # Higher chance of ice if snowing
            road_condition = weighted_choice(road_probs)
        else:
            road_condition = weighted_choice(road_conditions)
        
        # Light condition (based on time of day)
        if 6 <= hour < 18:  # Daytime
            light_probs = light_conditions.copy()
            light_probs['Daylight'] = 0.9
            light_condition = weighted_choice(light_probs)
        elif hour < 6 or hour >= 22:  # Night
            light_probs = light_conditions.copy()
            light_probs['Dark - lights on'] = 0.6
            light_probs['Dark - no lights'] = 0.35
            light_condition = weighted_choice(light_probs)
        else:  # Dawn/Dusk
            light_probs = light_conditions.copy()
            light_probs['Dawn/Dusk'] = 0.7
            light_condition = weighted_choice(light_probs)
        
        # Determine severity (correlated with road type, weather, etc.)
        base_severity = np.random.randint(1, 6)  # Base severity 1-5
        
        # Adjust based on factors
        severity_adjustment = 0
        
        # Road type affects severity
        if segment['road_type'] == 'primary':
            severity_adjustment += 0.5
        elif segment['road_type'] == 'residential':
            severity_adjustment -= 0.5
        
        # Weather affects severity
        if weather_condition in ['Rain', 'Snow', 'Fog']:
            severity_adjustment += 0.5
        
        # Road condition affects severity
        if road_condition in ['Ice', 'Snow']:
            severity_adjustment += 1
        elif road_condition == 'Wet':
            severity_adjustment += 0.5
        
        # Light condition affects severity
        if light_condition == 'Dark - no lights':
            severity_adjustment += 1
        elif light_condition == 'Dark - lights on':
            severity_adjustment += 0.5
        
        # Apply adjustment
        severity = max(1, min(5, int(base_severity + severity_adjustment)))
        
        # Create accident record
        accident = {
            'accident_id': f"SYN-{str(i).zfill(6)}",
            'timestamp': timestamp.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'severity': severity,
            'casualties': max(0, int(np.random.random() * (severity - 1) + np.random.random())),
            'vehicles_involved': np.random.randint(1, 4),  # 1-3 vehicles
            'weather_condition': weather_condition,
            'road_condition': road_condition,
            'light_condition': light_condition
        }
        
        accidents.append(accident)
    
    # Insert into database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Clear existing data
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
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Generated {len(accidents)} realistic accidents")
    return True

def import_real_data(city="Chicago", country="Illinois", max_roads=500, num_accidents=500):
    """
    Import real road network data and generate realistic synthetic accidents
    
    Args:
        city: City name
        country: Country or state name
        max_roads: Maximum number of road segments to import
        num_accidents: Maximum number of accidents to generate
    """
    print(f"\n=== IMPORTING REAL ROAD NETWORK FOR {city}, {country} ===\n")
    print(f"Limiting to {max_roads} roads and {num_accidents} accidents")
    
    # Set up database
    from database import setup_database
    setup_database()
    
    # Import road network
    success, road_segments = import_osm_road_network(city, country, max_roads=max_roads)
    
    if not success:
        print(f"Failed to import road network for {city}, {country}")
        return False
    
    # Generate realistic accidents on the road network
    generate_realistic_accidents(road_segments, num_accidents=num_accidents)
    
    print("\nReal road network with realistic synthetic accidents imported successfully!")
    return True


if __name__ == "__main__":
    # Default to Chicago if no arguments provided
    import sys
    
    city = sys.argv[1] if len(sys.argv) > 1 else "Chicago"
    country = sys.argv[2] if len(sys.argv) > 2 else "Illinois"
    max_roads = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    num_accidents = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    
    import_real_data(city, country, max_roads, num_accidents)

