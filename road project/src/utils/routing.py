import folium
import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString
import numpy as np

def get_safe_routes(start_coords, end_coords, hotspots, radius=0.2, output_file="output/maps/safe_route_map.html"):
    """
    Generate and visualize routes that avoid accident hotspots
    
    Args:
        start_coords: Tuple of (latitude, longitude) for start point
        end_coords: Tuple of (latitude, longitude) for end point
        hotspots: List of hotspot dictionaries from hotspot analysis
        radius: Distance to avoid around hotspots (in km)
        output_file: Where to save the map
    
    Returns:
        Path to the generated map
    """
    print(f"Generating safe route from {start_coords} to {end_coords}...")
    
    # Create bounding box
    north = max(start_coords[0], end_coords[0]) + 0.05
    south = min(start_coords[0], end_coords[0]) - 0.05
    east = max(start_coords[1], end_coords[1]) + 0.05
    west = min(start_coords[1], end_coords[1]) - 0.05
    
    # Create bbox dictionary for current osmnx version
    bbox = {"north": north, "south": south, "east": east, "west": west}
    
    # Get the road network graph
    print("Downloading road network...")
    G = ox.graph_from_bbox(**bbox, network_type='drive')
    
    # Convert hotspots to shapely points with buffer
    print("Processing hotspots...")
    hotspot_areas = []
    for hotspot in hotspots:
        # Create a circular area around each hotspot
        point = Point(hotspot['longitude'], hotspot['latitude'])
        
        # Different buffer size based on risk score if available
        buffer_radius = radius
        if 'risk_score' in hotspot:
            # Scale buffer by risk score, higher risk = larger avoidance area
            buffer_radius = radius * min(3, max(0.5, hotspot['risk_score'] / 10))
        
        # Convert to approx. km at this latitude
        # At equator, 0.01 degrees is about 1.11 km
        km_per_degree = 111.32  # km per degree at equator
        lat_correction = np.cos(np.radians(hotspot['latitude']))
        buffer_degrees = buffer_radius / (km_per_degree * lat_correction)
        
        # Create the buffered area
        buffered = point.buffer(buffer_degrees)
        hotspot_areas.append({
            'geometry': buffered,
            'risk_score': hotspot.get('risk_score', 1)
        })
    
    # Get the nearest nodes to our start and end points
    start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.nearest_nodes(G, end_coords[1], end_coords[0])
    
    # 1. First, calculate the standard shortest path (distance-based)
    print("Calculating standard route...")
    try:
        standard_route = nx.shortest_path(G, start_node, end_node, weight='length')
        standard_length = sum(ox.utils_graph.get_route_edge_attributes(G, standard_route, 'length'))
        print(f"Standard route length: {standard_length/1000:.2f} km")
    except nx.NetworkXNoPath:
        print("No standard route found between points")
        standard_route = None
        standard_length = 0
    
    # 2. Calculate a route that avoids hotspots
    print("Calculating safer route...")
    
    # Create a copy of the graph to modify edge weights
    G_safe = G.copy()
    
    # Add high weights to edges near hotspots
    for u, v, key, data in G_safe.edges(keys=True, data=True):
        # Get the midpoint of this edge
        if 'geometry' in data:
            # If we have the geometry, use the midpoint of the line
            mid_point = data['geometry'].interpolate(0.5, normalized=True)
            edge_x, edge_y = mid_point.x, mid_point.y
        else:
            # Otherwise calculate from node coordinates
            from_node = G_safe.nodes[u]
            to_node = G_safe.nodes[v]
            edge_x = (from_node['x'] + to_node['x']) / 2
            edge_y = (from_node['y'] + to_node['y']) / 2
        
        edge_point = Point(edge_x, edge_y)
        
        # Check if this edge is in any hotspot area
        for area in hotspot_areas:
            if area['geometry'].contains(edge_point):
                # Increase the weight of this edge to make it less attractive for routing
                # This effectively makes the route avoid this area if possible
                risk_multiplier = area.get('risk_score', 1) * 5  # Higher risk = higher avoidance
                G_safe[u][v][key]['length'] = data['length'] * risk_multiplier
                break
    
    # Calculate the safer route
    try:
        safe_route = nx.shortest_path(G_safe, start_node, end_node, weight='length')
        safe_length = sum(ox.utils_graph.get_route_edge_attributes(G, safe_route, 'length'))
        print(f"Safer route length: {safe_length/1000:.2f} km")
        
        # Calculate the safety difference
        if standard_length > 0:
            length_difference = (safe_length - standard_length) / 1000  # km
            percentage_increase = (safe_length / standard_length - 1) * 100
            print(f"Safety detour: +{length_difference:.2f} km (+{percentage_increase:.1f}%)")
        else:
            length_difference = 0
            percentage_increase = 0
    except nx.NetworkXNoPath:
        print("No safer route found between points")
        safe_route = None
        safe_length = 0
        length_difference = 0
        percentage_increase = 0
    
    # Create a map to visualize both routes
    m = folium.Map(location=start_coords, zoom_start=13)
    
    # Add start and end markers
    folium.Marker(
        location=start_coords,
        popup="Start",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        location=end_coords,
        popup="Destination",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add hotspots to the map
    hotspots_group = folium.FeatureGroup(name="Accident Hotspots")
    
    for i, hotspot in enumerate(hotspots):
        # Determine color based on risk score or cluster size
        if 'risk_score' in hotspot:
            risk = hotspot['risk_score']
            if risk > 20:
                color = 'red'
            elif risk > 10:
                color = 'orange'
            else:
                color = 'blue'
        else:
            # Based on accident count
            count = hotspot['accident_count']
            if count > 20:
                color = 'red'
            elif count > 10:
                color = 'orange'
            else:
                color = 'blue'
        
        # Circle size based on cluster size or buffer radius
        radius = hotspot['accident_count'] * 20
        
        # Add circle
        folium.Circle(
            location=[hotspot['latitude'], hotspot['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.4,
            popup=f"Hotspot #{i+1}: {hotspot['accident_count']} accidents, Risk: {hotspot.get('risk_score', 'N/A')}"
        ).add_to(hotspots_group)
    
    # Add standard route to map
    if standard_route:
        standard_route_group = folium.FeatureGroup(name="Standard Route")
        standard_route_coords = []
        for node in standard_route:
            node_data = G.nodes[node]
            standard_route_coords.append([node_data['y'], node_data['x']])
        
        folium.PolyLine(
            locations=standard_route_coords,
            color='blue',
            weight=4,
            opacity=0.7,
            popup=f"Standard Route: {standard_length/1000:.2f} km"
        ).add_to(standard_route_group)
        
        standard_route_group.add_to(m)
    
    # Add safer route to map
    if safe_route:
        safer_route_group = folium.FeatureGroup(name="Safer Route")
        safer_route_coords = []
        for node in safe_route:
            node_data = G.nodes[node]
            safer_route_coords.append([node_data['y'], node_data['x']])
        
        folium.PolyLine(
            locations=safer_route_coords,
            color='green',
            weight=4,
            opacity=0.7,
            popup=f"Safer Route: {safe_length/1000:.2f} km (+{percentage_increase:.1f}%)"
        ).add_to(safer_route_group)
        
        safer_route_group.add_to(m)
    
    # Add hotspots group
    hotspots_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend and info
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
               padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <h4>Route Comparison</h4>
        <p><span style="color:blue; font-weight:bold;">Standard Route:</span> {standard_length/1000:.2f} km</p>
        <p><span style="color:green; font-weight:bold;">Safer Route:</span> {safe_length/1000:.2f} km</p>
        <p>Safety Detour: +{length_difference:.2f} km (+{percentage_increase:.1f}%)</p>
        <hr>
        <p><i>The safer route avoids accident hotspots but may be slightly longer.</i></p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 9999; background-color: white; 
               padding: 10px; border: 2px solid grey; border-radius: 5px; text-align: center;">
        <h3>Safe Route Navigation Around Accident Hotspots</h3>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    m.save(output_file)
    print(f"Safe route map saved to {output_file}")
    
    return output_file
