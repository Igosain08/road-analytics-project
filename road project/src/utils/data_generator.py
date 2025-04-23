# src/utils/data_generator.py
import os
import json
import random
import math
import numpy as np
from datetime import datetime, timedelta

# Ensure output directories exist
os.makedirs('data/sample', exist_ok=True)

# City center coordinates (example: Chicago)
CITY_CENTER = {
    "lat": 41.8781,
    "lon": -87.6298
}

def random_coordinates(center_lat, center_lon, radius_km=10):
    """Generate random coordinates within a radius of the city center"""
    # Earth's radius in km
    R = 6371
    
    # Convert radius from km to degrees
    radius_lat = radius_km / R * (180 / math.pi)
    radius_lon = radius_km / (R * math.cos(center_lat * math.pi / 180)) * (180 / math.pi)
    
    # Random coordinates within the radius
    lat = center_lat + (random.random() * 2 - 1) * radius_lat
    lon = center_lon + (random.random() * 2 - 1) * radius_lon
    
    return {"lat": lat, "lon": lon}

def random_date(start, end):
    """Generate a random date between start and end dates"""
    delta = end - start
    random_days = random.randrange(delta.days)
    random_seconds = random.randrange(86400)  # seconds in a day
    return start + timedelta(days=random_days, seconds=random_seconds)

def weighted_random(items, weights):
    """Select item based on weights"""
    total_weight = sum(weights)
    random_val = random.random() * total_weight
    weight_sum = 0
    
    for i, weight in enumerate(weights):
        weight_sum += weight
        if random_val < weight_sum:
            return items[i]
    
    return items[0]  # fallback

def generate_accident_data(count=2000):
    """Generate sample accident data"""
    print(f"Generating {count} sample accident records...")
    
    accidents = []
    
    # Road conditions
    road_conditions = ['Dry', 'Wet', 'Snow', 'Ice', 'Slush', 'Unknown']
    road_condition_weights = [70, 15, 5, 5, 3, 2]  # percentages
    
    # Weather conditions
    weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Unknown']
    weather_condition_weights = [60, 20, 5, 5, 8, 2]  # percentages
    
    # Light conditions
    light_conditions = ['Daylight', 'Dark - lights on', 'Dark - no lights', 'Dawn/Dusk']
    light_condition_weights = [60, 20, 10, 10]  # percentages
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 years ago
    
    for i in range(count):
        coords = random_coordinates(CITY_CENTER["lat"], CITY_CENTER["lon"])
        timestamp = random_date(start_date, end_date)
        
        # More accidents during rush hours
        hour = timestamp.hour
        rush_hour = (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18)
        
        # Severity tends to be higher at night and in bad weather
        nighttime = hour < 6 or hour > 18
        weather_condition = weighted_random(weather_conditions, weather_condition_weights)
        bad_weather = weather_condition in ['Rain', 'Snow', 'Fog']
        
        if nighttime and bad_weather:
            severity = random.randint(3, 5)  # 3-5
        elif nighttime or bad_weather:
            severity = random.randint(2, 5)  # 2-5
        else:
            severity = random.randint(1, 5)  # 1-5
        
        light_condition = 'Dark - lights on' if nighttime else 'Daylight'
        if nighttime and random.random() < 0.3:  # 30% chance of no lights at night
            light_condition = 'Dark - no lights'
        if (hour >= 5 and hour < 7) or (hour >= 17 and hour < 19):  # Dawn/Dusk hours
            if random.random() < 0.7:  # 70% chance during these hours
                light_condition = 'Dawn/Dusk'
        
        accidents.append({
            "accident_id": f"ACC-{str(i).zfill(6)}",
            "timestamp": timestamp.isoformat(),
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "severity": severity,
            "casualties": max(0, int(random.random() * (severity - 1) + random.random())),
            "vehicles_involved": random.randint(1, 4),  # 1-4 vehicles
            "weather_condition": weather_condition,
            "road_condition": weighted_random(road_conditions, road_condition_weights),
            "light_condition": light_condition
        })
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1} accident records...")
    
    # Save to file
    output_file = "data/sample/accidents.json"
    with open(output_file, 'w') as f:
        json.dump(accidents, f, indent=2)
    
    print(f"Sample accident data saved to {output_file}")
    return accidents

def generate_road_network_data(area_size=20):
    """Generate sample road network data"""
    print("Generating sample road network data...")
    
    # Create a grid of roads around the city center
    road_segments = []
    road_types = ['primary', 'secondary', 'tertiary', 'residential']
    road_spacing = 0.005  # approximately 500m between roads
    
    # Calculate bounds (in degrees) around city center
    bounds = {
        "north": CITY_CENTER["lat"] + (area_size / 2) * 0.01,
        "south": CITY_CENTER["lat"] - (area_size / 2) * 0.01,
        "east": CITY_CENTER["lon"] + (area_size / 2) * 0.01,
        "west": CITY_CENTER["lon"] - (area_size / 2) * 0.01
    }
    
    segment_id = 1
    
    # Generate east-west roads
    for lat in np.arange(bounds["south"], bounds["north"], road_spacing):
        road_type = random.choice(road_types)
        speed_limit = {
            'primary': 50,
            'secondary': 40,
            'tertiary': 30,
            'residential': 25
        }[road_type]
        
        road_segments.append({
            "segment_id": f"RD-EW-{segment_id}",
            "start_lon": bounds["west"],
            "start_lat": lat,
            "end_lon": bounds["east"],
            "end_lat": lat,
            "road_type": road_type,
            "name": f"EW Road {segment_id // 2}",
            "lanes": 3 if road_type == 'primary' else 2 if road_type == 'secondary' else 1,
            "speed_limit": speed_limit,
            "oneway": 1 if random.random() < 0.2 else 0,  # 20% are one-way
            "surface": "asphalt" if random.random() < 0.8 else "concrete"
        })
        segment_id += 1
    
    # Generate north-south roads
    for lon in np.arange(bounds["west"], bounds["east"], road_spacing):
        road_type = random.choice(road_types)
        speed_limit = {
            'primary': 50,
            'secondary': 40,
            'tertiary': 30,
            'residential': 25
        }[road_type]
        
        road_segments.append({
            "segment_id": f"RD-NS-{segment_id}",
            "start_lon": lon,
            "start_lat": bounds["south"],
            "end_lon": lon,
            "end_lat": bounds["north"],
            "road_type": road_type,
            "name": f"NS Road {(segment_id - 1) // 2}",
            "lanes": 3 if road_type == 'primary' else 2 if road_type == 'secondary' else 1,
            "speed_limit": speed_limit,
            "oneway": 1 if random.random() < 0.2 else 0,  # 20% are one-way
            "surface": "asphalt" if random.random() < 0.8 else "concrete"
        })
        segment_id += 1
        
        if segment_id % 20 == 0:
            print(f"  Generated {segment_id} road segments...")
    
    # Save to file
    output_file = "data/sample/road_network.json"
    with open(output_file, 'w') as f:
        json.dump(road_segments, f, indent=2)
    
    print(f"Sample road network data with {len(road_segments)} segments saved to {output_file}")
    return road_segments

if __name__ == "__main__":
    print("\n=== GENERATING SAMPLE DATA ===\n")
    
    # Generate sample data
    accident_data = generate_accident_data(2000)
    road_data = generate_road_network_data()
    
    print("\nSample data generation complete!")
    print(f"- Generated {len(accident_data)} accident records")
    print(f"- Generated {len(road_data)} road segments")
