
# src/analysis/hotspot_analysis.py
import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from geopy.distance import geodesic
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

def identify_hotspots_simple(min_accidents=5, max_distance_km=0.5):
    """
    Identify accident hotspots using a simple distance-based approach
    
    Args:
        min_accidents: Minimum number of accidents to form a hotspot
        max_distance_km: Maximum distance (km) between accidents in a hotspot
        
    Returns:
        List of hotspots with location and accident count
    """
    print(f"Identifying hotspots (simple method, min_accidents={min_accidents}, max_distance={max_distance_km}km)...")
    
    # Get database connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all accidents
    cursor.execute("SELECT accident_id, latitude, longitude, severity FROM accidents")
    accidents = cursor.fetchall()
    
    # Convert to list of dictionaries
    accident_data = []
    for accident in accidents:
        accident_data.append({
            "accident_id": accident["accident_id"],
            "latitude": accident["latitude"],
            "longitude": accident["longitude"],
            "severity": accident["severity"],
            "cluster": None  # Will be assigned during clustering
        })
    
    # Simple clustering algorithm (similar to DBSCAN but simplified)
    cluster_id = 0
    
    for i, accident in enumerate(accident_data):
        # Skip if already assigned to a cluster
        if accident["cluster"] is not None:
            continue
        
        # Find neighboring accidents
        neighbors = []
        for j, other in enumerate(accident_data):
            if i == j:
                continue
            
            # Calculate distance between accidents
            distance = geodesic(
                (accident["latitude"], accident["longitude"]),
                (other["latitude"], other["longitude"])
            ).km
            
            if distance <= max_distance_km:
                neighbors.append(j)
        
        # If enough neighbors, form a cluster
        if len(neighbors) >= min_accidents - 1:  # -1 because we count the current accident
            cluster_id += 1
            accident["cluster"] = cluster_id
            
            # Assign neighbors to the same cluster
            for neighbor_idx in neighbors:
                accident_data[neighbor_idx]["cluster"] = cluster_id
    
    # Aggregate cluster information
    clusters = {}
    for accident in accident_data:
        if accident["cluster"] is not None:
            if accident["cluster"] not in clusters:
                clusters[accident["cluster"]] = {
                    "accident_count": 0,
                    "accident_ids": [],
                    "latitudes": [],
                    "longitudes": [],
                    "severities": []
                }
            
            clusters[accident["cluster"]]["accident_count"] += 1
            clusters[accident["cluster"]]["accident_ids"].append(accident["accident_id"])
            clusters[accident["cluster"]]["latitudes"].append(accident["latitude"])
            clusters[accident["cluster"]]["longitudes"].append(accident["longitude"])
            clusters[accident["cluster"]]["severities"].append(accident["severity"])
    
    # Calculate cluster centroids and average severity
    hotspots = []
    for cluster_id, cluster in clusters.items():
        if cluster["accident_count"] >= min_accidents:
            hotspots.append({
                "cluster_id": cluster_id,
                "accident_count": cluster["accident_count"],
                "latitude": sum(cluster["latitudes"]) / len(cluster["latitudes"]),
                "longitude": sum(cluster["longitudes"]) / len(cluster["longitudes"]),
                "avg_severity": sum(cluster["severities"]) / len(cluster["severities"])
            })
    
    # Sort hotspots by accident count (descending)
    hotspots.sort(key=lambda x: x["accident_count"], reverse=True)
    
    # Close connection
    conn.close()
    
    print(f"Identified {len(hotspots)} accident hotspots")
    return hotspots

def identify_hotspots_dbscan(eps_km=0.5, min_samples=5):
    """
    Identify accident hotspots using DBSCAN clustering algorithm
    
    Args:
        eps_km: The maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
        List of hotspots with location and accident count
    """
    print(f"Identifying hotspots (DBSCAN, eps={eps_km}km, min_samples={min_samples})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all accidents
    query = "SELECT accident_id, latitude, longitude, severity FROM accidents"
    df = pd.read_sql(query, conn)
    
    # Create a numpy array of coordinates
    coords = df[['latitude', 'longitude']].values
    
    # Define a custom function to convert degrees to kilometers
    def haversine_distance(a, b):
        return geodesic((a[0], a[1]), (b[0], b[1])).km
    
    # Apply DBSCAN
    db = DBSCAN(eps=eps_km, min_samples=min_samples, metric=haversine_distance, algorithm='ball_tree')
    df['cluster'] = db.fit_predict(coords)
    
    # Aggregate cluster information
    hotspots = []
    
    # Get all clusters (excluding noise points with cluster = -1)
    for cluster_id in sorted(df[df['cluster'] >= 0]['cluster'].unique()):
        cluster_points = df[df['cluster'] == cluster_id]
        
        hotspots.append({
            "cluster_id": int(cluster_id),
            "accident_count": len(cluster_points),
            "latitude": cluster_points['latitude'].mean(),
            "longitude": cluster_points['longitude'].mean(),
            "avg_severity": cluster_points['severity'].mean(),
            "accident_ids": cluster_points['accident_id'].tolist()
        })
    
    # Sort hotspots by accident count (descending)
    hotspots.sort(key=lambda x: x["accident_count"], reverse=True)
    
    # Close connection
    conn.close()
    
    print(f"Identified {len(hotspots)} accident hotspots using DBSCAN")
    return hotspots

def calculate_risk_factors():
    """
    Calculate risk factors from accident data
    
    Returns:
        Dictionary containing risk factor analysis results
    """
    print("Calculating risk factors...")
    
    # Get database connection
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total accident count
    cursor.execute("SELECT COUNT(*) FROM accidents")
    total_accidents = cursor.fetchone()[0]
    
    # Analyze weather conditions
    cursor.execute("""
        SELECT 
            weather_condition, 
            COUNT(*) as count,
            AVG(severity) as avg_severity,
            (COUNT(*) * 100.0 / ?) as percentage
        FROM accidents
        WHERE weather_condition IS NOT NULL
        GROUP BY weather_condition
        ORDER BY count DESC
    """, (total_accidents,))
    weather_factors = cursor.fetchall()
    
    # Analyze road conditions
    cursor.execute("""
        SELECT 
            road_condition, 
            COUNT(*) as count,
            AVG(severity) as avg_severity,
            (COUNT(*) * 100.0 / ?) as percentage
        FROM accidents
        WHERE road_condition IS NOT NULL
        GROUP BY road_condition
        ORDER BY count DESC
    """, (total_accidents,))
    road_factors = cursor.fetchall()
    
    # Analyze light conditions
    cursor.execute("""
        SELECT 
            light_condition, 
            COUNT(*) as count,
            AVG(severity) as avg_severity,
            (COUNT(*) * 100.0 / ?) as percentage
        FROM accidents
        WHERE light_condition IS NOT NULL
        GROUP BY light_condition
        ORDER BY count DESC
    """, (total_accidents,))
    light_factors = cursor.fetchall()
    
    # Analyze time patterns
    cursor.execute("""
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as count,
            AVG(severity) as avg_severity,
            (COUNT(*) * 100.0 / ?) as percentage
        FROM accidents
        GROUP BY hour
        ORDER BY hour
    """, (total_accidents,))
    hourly_patterns = cursor.fetchall()
    
    # Analyze day of week patterns
    cursor.execute("""
        SELECT 
            strftime('%w', timestamp) as day_of_week,
            COUNT(*) as count,
            AVG(severity) as avg_severity,
            (COUNT(*) * 100.0 / ?) as percentage
        FROM accidents
        GROUP BY day_of_week
        ORDER BY day_of_week
    """, (total_accidents,))
    daily_patterns = cursor.fetchall()
    
    # Close connection
    conn.close()
    
    # Return all factors
    risk_factors = {
        "total_accidents": total_accidents,
        "weather_factors": [dict(row) for row in weather_factors],
        "road_factors": [dict(row) for row in road_factors],
        "light_factors": [dict(row) for row in light_factors],
        "hourly_patterns": [dict(row) for row in hourly_patterns],
        "daily_patterns": [dict(row) for row in daily_patterns]
    }
    
    print("Risk factor analysis complete")
    return risk_factors

if __name__ == "__main__":
    print("\n=== HOTSPOT ANALYSIS ===\n")
    
    # Identify hotspots
    simple_hotspots = identify_hotspots_simple()
    dbscan_hotspots = identify_hotspots_dbscan()
    
    # Calculate risk factors
    risk_factors = calculate_risk_factors()
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total accidents analyzed: {risk_factors['total_accidents']}")
    print(f"Hotspots identified (simple method): {len(simple_hotspots)}")
    print(f"Hotspots identified (DBSCAN): {len(dbscan_hotspots)}")
    
    # Print top hotspots
    print("\nTop 5 hotspots (DBSCAN):")
    for i, hotspot in enumerate(dbscan_hotspots[:5], 1):
        print(f"  {i}. {hotspot['accident_count']} accidents, avg severity: {hotspot['avg_severity']:.2f}")

def identify_hotspots_hdbscan(min_cluster_size=10, min_samples=5, cluster_selection_epsilon=0.1):
    """
    Identify accident hotspots using HDBSCAN clustering algorithm
    
    Args:
        min_cluster_size: The minimum size of clusters (increase this value)
        min_samples: The number of samples in a neighborhood for a point to be considered a core point
        cluster_selection_epsilon: Controls the cluster merging threshold (increase for more distinct clusters)
        
    Returns:
        List of hotspots with location and accident count
    """
    print(f"Identifying hotspots (HDBSCAN, min_cluster_size={min_cluster_size}, min_samples={min_samples}, cluster_selection_epsilon={cluster_selection_epsilon})...")
    
    import hdbscan
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all accidents with additional features
    query = """
    SELECT accident_id, latitude, longitude, severity, 
           strftime('%H', timestamp) as hour,
           strftime('%w', timestamp) as day_of_week,
           weather_condition, road_condition, light_condition
    FROM accidents
    """
    df = pd.read_sql(query, conn)
    
    # Extract just the coordinates for clustering
    # This focuses purely on spatial clustering
    coords = df[['latitude', 'longitude']].values
    
    # Apply HDBSCAN with adjusted parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='haversine',
        gen_min_span_tree=True,
        prediction_data=True
    )
    
    df['cluster'] = clusterer.fit_predict(coords)
    
    # Get the outlier scores and cluster probabilities
    df['outlier_score'] = clusterer.outlier_scores_
    df['cluster_probability'] = clusterer.probabilities_
    
    # Count number of points assigned to noise (-1)
    noise_count = (df['cluster'] == -1).sum()
    total_count = len(df)
    noise_percentage = (noise_count / total_count) * 100
    print(f"Noise points: {noise_count} ({noise_percentage:.1f}% of total)")
    
    # Aggregate cluster information
    hotspots = []
    
    # Get all clusters (excluding noise points with cluster = -1)
    for cluster_id in sorted(df[df['cluster'] >= 0]['cluster'].unique()):
        cluster_points = df[df['cluster'] == cluster_id]
        
        # Calculate additional cluster metrics
        severity_distribution = cluster_points['severity'].value_counts().to_dict()
        
        # Extract dominant conditions
        dominant_conditions = {}
        for col in ['weather_condition', 'road_condition', 'light_condition']:
            if col in cluster_points.columns:
                try:
                    dominant_conditions[col.replace('_condition', '')] = cluster_points[col].mode()[0]
                except:
                    # In case mode can't be calculated
                    pass
        
        hotspots.append({
            "cluster_id": int(cluster_id),
            "accident_count": len(cluster_points),
            "latitude": cluster_points['latitude'].mean(),
            "longitude": cluster_points['longitude'].mean(),
            "avg_severity": cluster_points['severity'].mean(),
            "severity_distribution": severity_distribution,
            "dominant_conditions": dominant_conditions,
            "stability": cluster_points['cluster_probability'].mean(),
            "accident_ids": cluster_points['accident_id'].tolist()
        })
    
    # Sort hotspots by a combined risk score (count × avg_severity × stability)
    for hotspot in hotspots:
        hotspot["risk_score"] = hotspot["accident_count"] * hotspot["avg_severity"] * hotspot["stability"]
    
    hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
    
    conn.close()
    
    print(f"Identified {len(hotspots)} accident hotspots using HDBSCAN")
    return hotspots

def identify_spatiotemporal_hotspots(spatial_eps_km=0.5, temporal_eps_hours=24, min_samples=5):
    """
    Identify accident hotspots using ST-DBSCAN (Spatio-Temporal DBSCAN)
    
    Args:
        spatial_eps_km: Spatial threshold in kilometers
        temporal_eps_hours: Temporal threshold in hours
        min_samples: Minimum number of points to form a cluster
        
    Returns:
        List of hotspots with location, timeframe, and accident count
    """
    print(f"Identifying spatio-temporal hotspots...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all accidents
    query = "SELECT accident_id, latitude, longitude, timestamp, severity FROM accidents"
    df = pd.read_sql(query, conn)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract features for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Convert timestamps to hours since epoch for temporal distance
    df['hours_since_epoch'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1h')
    times = df['hours_since_epoch'].values.reshape(-1, 1)
    
    # Create a custom distance matrix that considers both space and time
    from sklearn.metrics import pairwise_distances
    
    def st_distance(X, Y):
        # Calculate spatial distances in km
        spatial_dist = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                spatial_dist[i, j] = geodesic((X[i][0], X[i][1]), (Y[j][0], Y[j][1])).km
        
        # Calculate temporal distances in hours
        temporal_dist = pairwise_distances(times, metric='euclidean')
        
        # Points are considered neighbors if they are close in BOTH space AND time
        neighbors = (spatial_dist <= spatial_eps_km) & (temporal_dist <= temporal_eps_hours)
        
        # Convert boolean matrix to distance matrix (0 for neighbors, infinity for non-neighbors)
        distance_matrix = np.where(neighbors, 0, np.inf)
        
        return distance_matrix
    
    # Apply DBSCAN with custom distance metric
    from sklearn.cluster import DBSCAN
    
    # For efficiency, limit to a smaller sample if dataset is large
    sample_size = min(1000, len(df))
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        coords = df[['latitude', 'longitude']].values
        times = df['hours_since_epoch'].values.reshape(-1, 1)
    
    # Calculate distance matrix
    dist_matrix = st_distance(coords, coords)
    
    # Fit DBSCAN
    db = DBSCAN(eps=0.5, min_samples=min_samples, metric='precomputed')
    df['cluster'] = db.fit_predict(dist_matrix)
    
    # Aggregate cluster information
    hotspots = []
    
    # Get all clusters (excluding noise points with cluster = -1)
    for cluster_id in sorted(df[df['cluster'] >= 0]['cluster'].unique()):
        cluster_points = df[df['cluster'] == cluster_id]
        
        # Calculate temporal characteristics of the cluster
        start_time = cluster_points['timestamp'].min()
        end_time = cluster_points['timestamp'].max()
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        hotspots.append({
            "cluster_id": int(cluster_id),
            "accident_count": len(cluster_points),
            "latitude": cluster_points['latitude'].mean(),
            "longitude": cluster_points['longitude'].mean(),
            "avg_severity": cluster_points['severity'].mean(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": duration_hours,
            "accident_ids": cluster_points['accident_id'].tolist()
        })
    
    # Sort hotspots by accident count (descending)
    hotspots.sort(key=lambda x: x["accident_count"], reverse=True)
    
    # Close connection
    conn.close()
    
    print(f"Identified {len(hotspots)} spatio-temporal accident hotspots")
    return hotspots
