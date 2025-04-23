import folium
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

# Ensure output directories exist
os.makedirs('output/maps', exist_ok=True)
os.makedirs('output/charts', exist_ok=True)

def create_accident_map(hotspots, output_file="output/maps/accident_map.html"):
    """Create an interactive map with accident locations and hotspots"""
    print(f"Creating interactive map ({output_file})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all accidents
    cursor = conn.cursor()
    cursor.execute("""
        SELECT latitude, longitude, severity, weather_condition, road_condition
        FROM accidents
        LIMIT 1000 -- Limit to 1000 to avoid browser performance issues
    """)
    accidents = cursor.fetchall()
    
    # Calculate map center (Chicago by default)
    cursor.execute("SELECT AVG(latitude), AVG(longitude) FROM accidents")
    center = cursor.fetchone()
    if center is None or center[0] is None or center[1] is None:
        center_lat, center_lon = 41.8781, -87.6298  # Chicago
    else:
        center_lat, center_lon = center[0], center[1]
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    
    # Add a title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Accident-Prone Road Areas Analysis</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create a feature group for accidents
    accidents_group = folium.FeatureGroup(name="Accidents")
    
    # Add accidents
    for accident in accidents:
        # Severity color scale
        severity_colors = {
            1: 'green',
            2: 'lightgreen',
            3: 'orange',
            4: 'red',
            5: 'darkred'
        }
        color = severity_colors.get(accident["severity"], 'blue')
        
        # Add circle marker
        folium.CircleMarker(
            location=[accident["latitude"], accident["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Severity: {accident['severity']}<br>Weather: {accident['weather_condition']}<br>Road: {accident['road_condition']}"
        ).add_to(accidents_group)
    
    # Add accidents group to map
    accidents_group.add_to(m)
    
    # Create a feature group for hotspots
    hotspots_group = folium.FeatureGroup(name="Hotspots")
    
    # Add hotspots
    for hotspot in hotspots:
        # Create a circle proportional to accident count
        radius = hotspot["accident_count"] * 20  # Scale by number of accidents
        
        folium.Circle(
            location=[hotspot["latitude"], hotspot["longitude"]],
            radius=radius,
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.2,
            popup=f"Hotspot #{hotspot['cluster_id']}<br>Accidents: {hotspot['accident_count']}<br>Avg Severity: {hotspot['avg_severity']:.2f}"
        ).add_to(hotspots_group)
    
    # Add hotspots group to map
    hotspots_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add map legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Severity Legend</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> 1 - Minor</p>
        <p><i class="fa fa-circle" style="color:lightgreen"></i> 2 - Moderate</p>
        <p><i class="fa fa-circle" style="color:orange"></i> 3 - Serious</p>
        <p><i class="fa fa-circle" style="color:red"></i> 4 - Severe</p>
        <p><i class="fa fa-circle" style="color:darkred"></i> 5 - Fatal</p>
        <p><i class="fa fa-circle" style="color:purple;opacity:0.3"></i> Hotspot</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map to file
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Close connection
    conn.close()
    
    return output_file

def create_road_segment_map(output_file="output/maps/road_network_map.html"):
    """Create an interactive map showing road segments with color coding by type"""
    print(f"Creating road network map ({output_file})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all road segments
    cursor = conn.cursor()
    cursor.execute("""
        SELECT segment_id, start_lat, start_lon, end_lat, end_lon, 
               road_type, name, speed_limit, lanes
        FROM road_segments
    """)
    segments = cursor.fetchall()
    
    # Calculate map center (Chicago by default)
    center_lat = sum([seg["start_lat"] + seg["end_lat"] for seg in segments]) / (2 * len(segments))
    center_lon = sum([seg["start_lon"] + seg["end_lon"] for seg in segments]) / (2 * len(segments))
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    
    # Add title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Road Network Analysis</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create feature groups for different road types
    primary_roads = folium.FeatureGroup(name="Primary Roads")
    secondary_roads = folium.FeatureGroup(name="Secondary Roads")
    tertiary_roads = folium.FeatureGroup(name="Tertiary Roads")
    residential_roads = folium.FeatureGroup(name="Residential Roads")
    
    # Road type colors
    road_colors = {
        'primary': '#FF0000',      # Red
        'secondary': '#FFA500',    # Orange
        'tertiary': '#FFFF00',     # Yellow
        'residential': '#808080'   # Gray
    }
    
    # Add road segments
    for segment in segments:
        # Create line for the road segment
        line = folium.PolyLine(
            locations=[[segment["start_lat"], segment["start_lon"]], 
                       [segment["end_lat"], segment["end_lon"]]],
            color=road_colors.get(segment["road_type"], 'blue'),
            weight=2 + segment["lanes"],  # Width based on number of lanes
            popup=f"{segment['name']}<br>Type: {segment['road_type']}<br>Speed Limit: {segment['speed_limit']} mph<br>Lanes: {segment['lanes']}"
        )
        
        # Add to the appropriate feature group
        if segment["road_type"] == "primary":
            line.add_to(primary_roads)
        elif segment["road_type"] == "secondary":
            line.add_to(secondary_roads)
        elif segment["road_type"] == "tertiary":
            line.add_to(tertiary_roads)
        else:
            line.add_to(residential_roads)
    
    # Add feature groups to map
    primary_roads.add_to(m)
    secondary_roads.add_to(m)
    tertiary_roads.add_to(m)
    residential_roads.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Road Types</b></p>
        <p><i class="fa fa-square" style="color:#FF0000"></i> Primary</p>
        <p><i class="fa fa-square" style="color:#FFA500"></i> Secondary</p>
        <p><i class="fa fa-square" style="color:#FFFF00"></i> Tertiary</p>
        <p><i class="fa fa-square" style="color:#808080"></i> Residential</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map to file
    m.save(output_file)
    print(f"Road network map saved to {output_file}")
    
    # Close connection
    conn.close()
    
    return output_file

def create_heatmap(output_file="output/maps/accident_heatmap.html"):
    """Create a heatmap showing accident density"""
    print(f"Creating accident heatmap ({output_file})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all accidents
    query = "SELECT latitude, longitude, severity FROM accidents"
    df = pd.read_sql(query, conn)
    
    # Calculate map center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
    
    # Add title
    title_html = '''
        <h3 align="center" style="font-size:16px; color:white;"><b>Accident Density Heatmap</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create heatmap data list with severity as weight
    heat_data = [[row['latitude'], row['longitude'], row['severity']] for index, row in df.iterrows()]
    
    # Add HeatMap layer
    HeatMap(heat_data, 
            radius=15, 
            blur=10, 
            gradient={
                '0.4': 'blue', 
                '0.65': 'lime', 
                '0.8': 'yellow', 
                '1': 'red'
            }).add_to(m)
    
    # Save map to file
    m.save(output_file)
    print(f"Heatmap saved to {output_file}")
    
    # Close connection
    conn.close()
    
    return output_file

def create_risk_factor_visualizations(risk_factors, output_file="output/charts/risk_factors.png"):
    """Create visualizations for risk factors"""
    print(f"Creating risk factor visualizations ({output_file})...")
    
    # Set up the figure
    plt.figure(figsize=(18, 12))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Weather Conditions
    plt.subplot(2, 3, 1)
    weather_df = pd.DataFrame(risk_factors["weather_factors"])
    if not weather_df.empty:
        sns.barplot(x='weather_condition', y='percentage', data=weather_df, palette='viridis')
        plt.title('Accidents by Weather Condition', fontsize=14)
        plt.ylabel('Percentage of Accidents', fontsize=12)
        plt.xlabel('Weather Condition', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Road Conditions
    plt.subplot(2, 3, 2)
    road_df = pd.DataFrame(risk_factors["road_factors"])
    if not road_df.empty:
        sns.barplot(x='road_condition', y='percentage', data=road_df, palette='viridis')
        plt.title('Accidents by Road Condition', fontsize=14)
        plt.ylabel('Percentage of Accidents', fontsize=12)
        plt.xlabel('Road Condition', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Light Conditions
    plt.subplot(2, 3, 3)
    light_df = pd.DataFrame(risk_factors["light_factors"])
    if not light_df.empty:
        sns.barplot(x='light_condition', y='percentage', data=light_df, palette='viridis')
        plt.title('Accidents by Light Condition', fontsize=14)
        plt.ylabel('Percentage of Accidents', fontsize=12)
        plt.xlabel('Light Condition', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hourly Pattern
    plt.subplot(2, 3, 4)
    hour_df = pd.DataFrame(risk_factors["hourly_patterns"])
    if not hour_df.empty:
        hour_df['hour'] = hour_df['hour'].astype(int)
        hour_df = hour_df.sort_values('hour')
        
        plt.plot(hour_df['hour'], hour_df['count'], marker='o', linewidth=2, color='#3498db')
        plt.fill_between(hour_df['hour'], hour_df['count'], alpha=0.2, color='#3498db')
        plt.title('Accidents by Hour of Day', fontsize=14)
        plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
        plt.ylabel('Number of Accidents', fontsize=12)
        plt.xticks(range(0, 24, 2), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for peak hours
        peak_hours = hour_df.nlargest(2, 'count')
        for _, peak in peak_hours.iterrows():
            plt.annotate(f"{int(peak['count'])}",
                        (peak['hour'], peak['count']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontweight='bold')
    
    # Daily Pattern
    plt.subplot(2, 3, 5)
    day_df = pd.DataFrame(risk_factors["daily_patterns"])
    if not day_df.empty:
        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        day_df['day_name'] = day_df['day_of_week'].astype(int).apply(lambda x: day_names[x])
        # Sort by day of week
        day_df['sort_order'] = day_df['day_of_week'].astype(int)
        day_df = day_df.sort_values('sort_order')
        
        sns.barplot(x='day_name', y='count', data=day_df, palette='viridis')
        plt.title('Accidents by Day of Week', fontsize=14)
        plt.ylabel('Number of Accidents', fontsize=12)
        plt.xlabel('Day of Week', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Severity by Weather
    plt.subplot(2, 3, 6)
    if not weather_df.empty:
        sns.barplot(x='weather_condition', y='avg_severity', data=weather_df, palette='rocket')
        plt.title('Average Severity by Weather Condition', fontsize=14)
        plt.ylabel('Average Severity (1-5)', fontsize=12)
        plt.xlabel('Weather Condition', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a title to the entire figure
    plt.suptitle('Accident Risk Factor Analysis', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Risk factor visualizations saved to {output_file}")
    plt.close()
    
    return output_file

def create_severity_distribution_chart(output_file="output/charts/severity_distribution.png"):
    """Create a chart showing the distribution of accident severities"""
    print(f"Creating severity distribution chart ({output_file})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get severity distribution
    query = """
    SELECT severity, COUNT(*) as count
    FROM accidents
    GROUP BY severity
    ORDER BY severity
    """
    df = pd.read_sql(query, conn)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Bar plot with gradient colors
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']  # Green to purple
    bars = plt.bar(df['severity'], df['count'], color=colors[:len(df)])
    
    # Add labels and title
    plt.title('Distribution of Accident Severities', fontsize=16)
    plt.xlabel('Severity Level', fontsize=14)
    plt.ylabel('Number of Accidents', fontsize=14)
    
    # Add severity level descriptions
    severity_labels = {
        1: 'Minor',
        2: 'Moderate',
        3: 'Serious',
        4: 'Severe',
        5: 'Fatal'
    }
    
    plt.xticks(df['severity'], [f"{s} - {severity_labels.get(s, '')}" for s in df['severity']], fontsize=12)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:,}',
                 ha='center', va='bottom', fontsize=12)
    
    # Add percentage labels inside bars
    total = df['count'].sum()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{percentage:.1f}%',
                 ha='center', va='center', fontsize=12, 
                 color='white', fontweight='bold')
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Severity distribution chart saved to {output_file}")
    plt.close()
    
    # Close connection
    conn.close()
    
    return output_file

def create_time_series_analysis(output_file="output/charts/time_series.png"):
    """Create a time series analysis of accidents over time"""
    print(f"Creating time series analysis ({output_file})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get monthly accident counts
    query = """
    SELECT 
        strftime('%Y-%m', timestamp) as month,
        COUNT(*) as count,
        AVG(severity) as avg_severity
    FROM accidents
    GROUP BY month
    ORDER BY month
    """
    df = pd.read_sql(query, conn)
    
    # Convert month to datetime for better plotting
    df['month'] = pd.to_datetime(df['month'] + '-01')
    
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # First subplot: Accident counts over time
    ax1.plot(df['month'], df['count'], marker='o', linestyle='-', linewidth=2, color='#3498db')
    ax1.set_title('Monthly Accident Counts', fontsize=14)
    ax1.set_ylabel('Number of Accidents', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line (simple moving average)
    window_size = min(5, len(df))  # Use 5 months or less if we have fewer data points
    if len(df) >= window_size:
        df['moving_avg'] = df['count'].rolling(window=window_size, center=True).mean()
        ax1.plot(df['month'], df['moving_avg'], 'r--', linewidth=2, label=f'{window_size}-Month Moving Average')
        ax1.legend(fontsize=12)
    
    # Second subplot: Average severity over time
    ax2.plot(df['month'], df['avg_severity'], marker='s', linestyle='-', linewidth=2, color='#e74c3c')
    ax2.set_title('Monthly Average Accident Severity', fontsize=14)
    ax2.set_ylabel('Average Severity (1-5)', fontsize=12)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits for severity (1-5 range with some padding)
    ax2.set_ylim(max(0.5, df['avg_severity'].min() - 0.5), min(5.5, df['avg_severity'].max() + 0.5))
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add overall title
    plt.suptitle('Accident Trends Over Time', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Time series analysis saved to {output_file}")
    plt.close()
    
    # Close connection
    conn.close()
    
    return output_file

def create_hdbscan_hotspot_map(hotspots, output_file="output/maps/hdbscan_hotspot_map.html"):
    """Create an interactive map with HDBSCAN hotspots"""
    print(f"Creating HDBSCAN hotspot map ({output_file})...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Calculate map center
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(latitude), AVG(longitude) FROM accidents")
    center = cursor.fetchone()
    if center is None or center[0] is None or center[1] is None:
        center_lat, center_lon = 41.8781, -87.6298  # Chicago
    else:
        center_lat, center_lon = center[0], center[1]
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    
    # Add a title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Enhanced Accident Hotspot Analysis (HDBSCAN)</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create feature groups for different risk levels
    high_risk = folium.FeatureGroup(name="High Risk Hotspots")
    medium_risk = folium.FeatureGroup(name="Medium Risk Hotspots")
    low_risk = folium.FeatureGroup(name="Low Risk Hotspots")
    
    # Calculate max accident count for scaling
    max_accident_count = max([h["accident_count"] for h in hotspots]) if hotspots else 1
    
    # Add hotspots
    for hotspot in hotspots:
        # Scale circle radius based on relative accident count (not absolute)
        # This prevents huge circles
        relative_size = hotspot["accident_count"] / max_accident_count
        base_radius = 100 + (relative_size * 300)  # Base between 100-400 meters
        
        # Apply risk score multiplier if available
        if "risk_score" in hotspot:
            risk_score = hotspot["risk_score"]
            # Normalize risk score to prevent extremely large circles
            risk_multiplier = min(1.5, max(0.5, risk_score / (max_accident_count * 5)))
            radius = base_radius * risk_multiplier
        else:
            radius = base_radius
        
        # Determine color based on risk score or accident count
        if "risk_score" in hotspot:
            risk_score = hotspot["risk_score"]
            max_risk = max([h.get("risk_score", 0) for h in hotspots])
            relative_risk = risk_score / max_risk if max_risk > 0 else 0
            
            if relative_risk > 0.7:
                color = 'red'
                group = high_risk
            elif relative_risk > 0.4:
                color = 'orange'
                group = medium_risk
            else:
                color = 'blue'
                group = low_risk
        else:
            # Fallback to accident count if risk score not available
            if relative_size > 0.7:
                color = 'red'
                group = high_risk
            elif relative_size > 0.4:
                color = 'orange'
                group = medium_risk
            else:
                color = 'blue'
                group = low_risk
        
        # Create popup content with detailed information
        popup_html = f"""
        <div style="width: 250px">
            <h4>Hotspot #{hotspot['cluster_id']}</h4>
            <b>Accidents:</b> {hotspot['accident_count']}<br>
            <b>Avg Severity:</b> {hotspot['avg_severity']:.2f}<br>
        """
        
        if "risk_score" in hotspot:
            popup_html += f"<b>Risk Score:</b> {hotspot['risk_score']:.2f}<br>"
        
        if "stability" in hotspot:
            popup_html += f"<b>Cluster Stability:</b> {hotspot['stability']:.2f}<br>"
        
        # Add dominant conditions if available
        if 'dominant_conditions' in hotspot:
            popup_html += "<h5>Dominant Conditions:</h5>"
            for condition_type, value in hotspot['dominant_conditions'].items():
                popup_html += f"<b>{condition_type.capitalize()}:</b> {value}<br>"
        
        # Add severity distribution if available
        if 'severity_distribution' in hotspot:
            popup_html += "<h5>Severity Distribution:</h5>"
            for severity, count in hotspot['severity_distribution'].items():
                popup_html += f"Level {severity}: {count} accidents<br>"
        
        popup_html += "</div>"
        
        # Create the circle marker for this hotspot
        folium.Circle(
            location=[hotspot["latitude"], hotspot["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(group)
    
    # Add all accidents as small dots for context
    accidents_group = folium.FeatureGroup(name="All Accidents")
    
    # Get all accidents
    accidents_df = pd.read_sql("SELECT latitude, longitude, severity FROM accidents LIMIT 1000", conn)
    
    # Add markers for accidents
    for _, accident in accidents_df.iterrows():
        folium.CircleMarker(
            location=[accident["latitude"], accident["longitude"]],
            radius=2,
            color='gray',
            fill=True,
            fill_opacity=0.5,
            weight=1
        ).add_to(accidents_group)
    
    # Add feature groups to map in specific order
    accidents_group.add_to(m)
    low_risk.add_to(m)
    medium_risk.add_to(m)
    high_risk.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Risk Levels</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> High Risk</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium Risk</p>
        <p><i class="fa fa-circle" style="color:blue"></i> Low Risk</p>
        <p><i class="fa fa-circle" style="color:gray;opacity:0.5"></i> Individual Accidents</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map to file
    m.save(output_file)
    print(f"HDBSCAN hotspot map saved to {output_file}")
    
    # Close connection
    conn.close()
    
    return output_file

def create_predictive_risk_map(output_file="output/maps/predictive_risk_map.html"):
    """Create a map showing predicted accident risk for road segments"""
    print(f"Creating predictive risk map ({output_file})...")
    
    # Import prediction function
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from analysis.prediction_models import predict_accident_probability
    except ImportError as e:
        print(f"Error importing prediction model: {e}")
        print("Make sure you've implemented the prediction_models.py file")
        return None
    
    # Get database connection
    conn = get_db_connection()
    
    # Get all road segments
    query = "SELECT segment_id, start_lat, start_lon, end_lat, end_lon, road_type, name FROM road_segments"
    road_segments = pd.read_sql(query, conn)
    
    # Calculate map center
    center_lat = road_segments[['start_lat', 'end_lat']].values.flatten().mean()
    center_lon = road_segments[['start_lon', 'end_lon']].values.flatten().mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    
    # Add title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Predictive Road Risk Map</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create feature groups for different risk levels
    high_risk = folium.FeatureGroup(name="High Risk Roads")
    medium_risk = folium.FeatureGroup(name="Medium Risk Roads")
    low_risk = folium.FeatureGroup(name="Low Risk Roads")
    
    # Current conditions (can be modified to test different scenarios)
    current_conditions = {
        'hour': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'weather_condition': 'Clear',
        'road_condition': 'Dry',
        'light_condition': 'Daylight' if 6 <= datetime.now().hour <= 18 else 'Dark - lights on'
    }
    
    # Process a subset of road segments for demonstration (to save time)
    sample_size = min(100, len(road_segments))
    
    # Create a sample with representation from different road types
    samples = []
    for road_type in road_segments['road_type'].unique():
        type_segments = road_segments[road_segments['road_type'] == road_type]
        if len(type_segments) > 0:
            type_sample = type_segments.sample(min(25, len(type_segments)))
            samples.append(type_sample)
    
    sampled_roads = pd.concat(samples) if samples else road_segments.sample(sample_size)
    
    # Track how many successful predictions we've made
    successful_predictions = 0
    max_attempts = min(200, len(road_segments))
    attempts = 0
    
    while successful_predictions < sample_size and attempts < max_attempts:
        # Get a random road segment
        segment = road_segments.sample(1).iloc[0]
        attempts += 1
        
        # Try to predict risk for this segment
        try:
            prediction = predict_accident_probability(segment['segment_id'], current_conditions)
            
            if 'error' in prediction:
                print(f"Error for segment {segment['segment_id']}: {prediction['error']}")
                continue
                
            probability = prediction['accident_probability']
            risk_level = prediction['risk_level']
            
            # Determine color based on risk level
            if risk_level in ['High', 'Severe']:
                color = 'red'
                width = 4
                group = high_risk
            elif risk_level in ['Elevated', 'Moderate']:
                color = 'orange'
                width = 3
                group = medium_risk
            else:
                color = 'green'
                width = 2
                group = low_risk
            
            # Create popup content
            popup_html = f"""
            <div style="width: 250px">
                <h4>{segment['name']}</h4>
                <b>Road Type:</b> {segment['road_type']}<br>
                <b>Risk Level:</b> {risk_level}<br>
                <b>Accident Probability:</b> {probability:.4f}<br>
                <h5>Current Conditions:</h5>
                <b>Weather:</b> {current_conditions['weather_condition']}<br>
                <b>Road:</b> {current_conditions['road_condition']}<br>
                <b>Light:</b> {current_conditions['light_condition']}<br>
                <b>Time:</b> {current_conditions['hour']}:00<br>
            """
            
            # Add top risk factors if available
            if 'top_risk_factors' in prediction:
                popup_html += "<h5>Top Risk Factors:</h5>"
                for factor, value in list(prediction['top_risk_factors'].items())[:3]:
                    popup_html += f"<b>{factor}:</b> {value:.4f}<br>"
            
            # Add mock indicator if applicable
            if prediction.get('is_mock', False):
                popup_html += "<p><i>(Using estimated risk values)</i></p>"
                
            popup_html += "</div>"
            
            # Create the line for this road segment
            folium.PolyLine(
                locations=[[segment['start_lat'], segment['start_lon']], 
                           [segment['end_lat'], segment['end_lon']]],
                color=color,
                weight=width,
                opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(group)
            
            successful_predictions += 1
            
        except Exception as e:
            print(f"Error predicting for segment {segment['segment_id']}: {e}")
    
    print(f"Successfully visualized {successful_predictions} road segments")
    
    # Add feature groups to map
    high_risk.add_to(m)
    medium_risk.add_to(m)
    low_risk.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Predicted Risk</b></p>
        <p><i class="fa fa-minus" style="color:red"></i> High/Severe Risk</p>
        <p><i class="fa fa-minus" style="color:orange"></i> Moderate/Elevated Risk</p>
        <p><i class="fa fa-minus" style="color:green"></i> Low Risk</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add note about model status
    if not os.path.exists('models/accident_probability_model.joblib'):
        note_html = '''
        <div style="position: fixed; bottom: 150px; right: 50px; z-index: 1000; background-color: #fff8e1; padding: 10px; border: 2px solid #ffcc80; border-radius: 5px;">
            <p><b>Note:</b> Using estimated risk values. Train prediction models to get actual predictions.</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(note_html))
    
    # Save map to file
    m.save(output_file)
    print(f"Predictive risk map saved to {output_file}")
    
    # Close connection
    conn.close()
    
    return output_file


if __name__ == "__main__":
    print("\n=== VISUALIZATION GENERATION ===\n")
    
    # Import analysis modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.hotspot_analysis import identify_hotspots_dbscan, identify_hotspots_hdbscan, calculate_risk_factors
    
    # Get hotspots using different methods
    dbscan_hotspots = identify_hotspots_dbscan()
    
    # Try to get HDBSCAN hotspots if available
    try:
        hdbscan_hotspots = identify_hotspots_hdbscan()
        print(f"Identified {len(hdbscan_hotspots)} hotspots using HDBSCAN")
    except ImportError:
        print("HDBSCAN not available. Install with: pip install hdbscan")
        hdbscan_hotspots = []
    
    # Get risk factors
    risk_factors = calculate_risk_factors()
    
    # Create basic visualizations
    create_accident_map(dbscan_hotspots)
    create_road_segment_map()
    create_heatmap()
    create_risk_factor_visualizations(risk_factors)
    create_severity_distribution_chart()
    create_time_series_analysis()
    
    # Create enhanced visualizations
    if hdbscan_hotspots:
        create_hdbscan_hotspot_map(hdbscan_hotspots)
    
    # Create predictive visualizations
    try:
        create_predictive_risk_map()
    except Exception as e:
        print(f"Warning: Could not create predictive risk map: {e}")
        print("Run the prediction model training first with 'python src/analysis/prediction_models.py'")
    
    print("\nAll visualizations created successfully!")
