import os
import sys
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection
from analysis.hotspot_analysis import identify_hotspots_hdbscan

# Create output directory
os.makedirs('output/dashboard', exist_ok=True)

def create_interactive_dashboard():
    """
    Create an interactive web dashboard for exploring accident data
    """
    print("Initializing interactive dashboard...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Load data
    accidents_df = pd.read_sql("SELECT * FROM accidents", conn)
    road_segments_df = pd.read_sql("SELECT * FROM road_segments", conn)
    
    # Convert timestamp to datetime
    accidents_df['timestamp'] = pd.to_datetime(accidents_df['timestamp'])
    accidents_df['hour'] = accidents_df['timestamp'].dt.hour
    accidents_df['day_of_week'] = accidents_df['timestamp'].dt.dayofweek
    accidents_df['month'] = accidents_df['timestamp'].dt.month
    accidents_df['year'] = accidents_df['timestamp'].dt.year
    
    # Create day name mapping
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    accidents_df['day_name'] = accidents_df['day_of_week'].map(day_names)
    
    # Get map center coordinates
    center_lat = accidents_df['latitude'].mean()
    center_lon = accidents_df['longitude'].mean()
    
    # Get min/max coordinates
    min_lat = accidents_df['latitude'].min()
    max_lat = accidents_df['latitude'].max()
    min_lon = accidents_df['longitude'].min()
    max_lon = accidents_df['longitude'].max()
    
    # Close database connection
    conn.close()
    
    # Create Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Define app layout with tabs
    app.layout = html.Div([
        html.H1("Road Safety Analysis Dashboard", style={'textAlign': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
        
        # Add tabs for different dashboard views
        dcc.Tabs([
            # Tab 1: Data Exploration
            dcc.Tab(label='Data Exploration', children=[
                html.Div([
                    html.Div([
                        html.H3("Filters", style={'textAlign': 'center'}),
                        
                        html.Label("Severity Range:"),
                        dcc.RangeSlider(
                            id='severity-slider',
                            min=1,
                            max=5,
                            step=1,
                            marks={i: str(i) for i in range(1, 6)},
                            value=[1, 5]
                        ),
                        
                        html.Label("Date Range:", style={'margin-top': '15px'}),
                        dcc.DatePickerRange(
                            id='date-picker',
                            start_date=accidents_df['timestamp'].min().date(),
                            end_date=accidents_df['timestamp'].max().date(),
                            max_date_allowed=accidents_df['timestamp'].max().date(),
                            min_date_allowed=accidents_df['timestamp'].min().date()
                        ),
                        
                        html.Label("Weather Condition:", style={'margin-top': '15px'}),
                        dcc.Dropdown(
                            id='weather-dropdown',
                            options=[{'label': cond, 'value': cond} 
                                    for cond in sorted(accidents_df['weather_condition'].unique()) if cond],
                            value=[],
                            multi=True
                        ),
                        
                        html.Label("Road Condition:", style={'margin-top': '15px'}),
                        dcc.Dropdown(
                            id='road-dropdown',
                            options=[{'label': cond, 'value': cond} 
                                    for cond in sorted(accidents_df['road_condition'].unique()) if cond],
                            value=[],
                            multi=True
                        ),
                        
                        html.Label("Time of Day:", style={'margin-top': '15px'}),
                        dcc.RangeSlider(
                            id='time-slider',
                            min=0,
                            max=23,
                            step=1,
                            marks={i: str(i) for i in range(0, 24, 4)},
                            value=[0, 23]
                        ),
                        
                        html.Button(
                            'Apply Filters', 
                            id='apply-filters', 
                            n_clicks=0,
                            style={'margin-top': '20px', 'width': '100%', 'padding': '10px'}
                        ),
                        
                        html.Div(id='filter-summary', style={'margin-top': '15px', 'font-style': 'italic'})
                        
                    ], style={'width': '25%', 'padding': '20px', 'background': '#f0f0f0', 'border-radius': '10px'}),
                    
                    html.Div([
                        html.H3("Accident Map", style={'textAlign': 'center'}),
                        dcc.Graph(id='accident-map', style={'height': '70vh'})
                    ], style={'width': '70%', 'padding': '20px'})
                    
                ], style={'display': 'flex', 'gap': '20px', 'margin-bottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H3("Accidents by Severity", style={'textAlign': 'center'}),
                        dcc.Graph(id='severity-chart')
                    ], style={'width': '48%', 'padding': '20px', 'background': '#fff', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
                    
                    html.Div([
                        html.H3("Accidents by Time of Day", style={'textAlign': 'center'}),
                        dcc.Graph(id='time-chart')
                    ], style={'width': '48%', 'padding': '20px', 'background': '#fff', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H3("Accidents by Day of Week", style={'textAlign': 'center'}),
                        dcc.Graph(id='day-chart')
                    ], style={'width': '48%', 'padding': '20px', 'background': '#fff', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)'}),
                    
                    html.Div([
                        html.H3("Accidents by Weather Condition", style={'textAlign': 'center'}),
                        dcc.Graph(id='weather-chart')
                    ], style={'width': '48%', 'padding': '20px', 'background': '#fff', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
            ]),
            
            # Tab 2: Hotspot Analysis
            dcc.Tab(label='Hotspot Analysis', children=[
                html.Div([
                    html.H3("Accident Hotspots", style={'textAlign': 'center'}),
                    html.P("This map shows the identified accident hotspots with risk levels.", 
                          style={'textAlign': 'center'}),
                    
                    # Embedded hotspot map
                    html.Iframe(
                        id='hotspot-iframe',
                        src='/output/maps/hdbscan_hotspot_map.html',
                        style={'width': '100%', 'height': '700px', 'border': 'none', 
                               'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)'}
                    )
                ], style={'padding': '20px'})
            ]),
            
            # Tab 3: Risk Prediction
            dcc.Tab(label='Risk Prediction', children=[
                html.Div([
                    html.H3("Road Segment Risk Analysis", style={'textAlign': 'center'}),
                    html.P("This map shows the predicted risk level for different road segments under current conditions.", 
                          style={'textAlign': 'center'}),
                    
                    # Embedded risk prediction map
                    html.Iframe(
                        id='risk-map-iframe',
                        src='/output/maps/predictive_risk_map.html',
                        style={'width': '100%', 'height': '700px', 'border': 'none', 
                               'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0,0,0,0.1)'}
                    )
                ], style={'padding': '20px'})
            ]),
            
            # Tab 4: Safe Route Planning
            dcc.Tab(label='Safe Route Planning', children=[
                html.Div([
                    html.H3("Plan a Safe Route", style={'textAlign': 'center'}),
                    html.P("Generate a route that avoids accident hotspots", style={'textAlign': 'center'}),
                    
                    html.Div([
                        html.Div([
                            html.Label("Start Point (latitude, longitude):"),
                            dcc.Input(
                                id='start-coords',
                                type='text',
                                value=f"{center_lat:.6f}, {center_lon:.6f}",
                                style={'width': '100%', 'marginBottom': '10px'}
                            ),
                            
                            html.Label("End Point (latitude, longitude):"),
                            dcc.Input(
                                id='end-coords',
                                type='text',
                                value=f"{center_lat + 0.01:.6f}, {center_lon + 0.01:.6f}",
                                style={'width': '100%', 'marginBottom': '20px'}
                            ),
                            
                            html.Button(
                                'Generate Safe Route', 
                                id='generate-route-button',
                                n_clicks=0,
                                style={'width': '100%', 'padding': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}
                            ),
                            
                            html.Div(id='route-info', style={'marginTop': '20px'})
                        ], style={'width': '30%', 'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '10px'}),
                        
                        html.Div([
                            html.Div(id='route-map-container', style={'height': '600px', 'width': '100%'})
                        ], style={'width': '65%', 'padding': '20px'})
                    ], style={'display': 'flex', 'gap': '20px'})
                ])
            ]),
        ], style={'margin-bottom': '20px'}),
        
        # Store the filtered dataframe
        dcc.Store(id='filtered-data')
    ], style={'max-width': '1400px', 'margin': '0 auto', 'padding': '0 20px'})
    
    # Define callbacks
    @app.callback(
        Output('filtered-data', 'data'),
        [Input('apply-filters', 'n_clicks')],
        [State('severity-slider', 'value'),
         State('date-picker', 'start_date'),
         State('date-picker', 'end_date'),
         State('weather-dropdown', 'value'),
         State('road-dropdown', 'value'),
         State('time-slider', 'value')]
    )
    def filter_data(n_clicks, severity_range, start_date, end_date, weather_conditions, road_conditions, time_range):
        # Create a copy of the original dataframe
        filtered_df = accidents_df.copy()
        
        # Apply filters
        filtered_df = filtered_df[
            (filtered_df['severity'] >= severity_range[0]) & 
            (filtered_df['severity'] <= severity_range[1])
        ]
        
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['timestamp'] >= start_date) & 
                (filtered_df['timestamp'] <= end_date)
            ]
        
        if weather_conditions:
            filtered_df = filtered_df[filtered_df['weather_condition'].isin(weather_conditions)]
        
        if road_conditions:
            filtered_df = filtered_df[filtered_df['road_condition'].isin(road_conditions)]
        
        filtered_df = filtered_df[
            (filtered_df['hour'] >= time_range[0]) & 
            (filtered_df['hour'] <= time_range[1])
        ]
        
        # Return the filtered dataframe as JSON
        return filtered_df.to_json(date_format='iso', orient='split')
    
    @app.callback(
        Output('filter-summary', 'children'),
        [Input('filtered-data', 'data')]
    )
    def update_filter_summary(json_data):
        if not json_data:
            return "No data selected"
        
        # Parse the JSON data
        filtered_df = pd.read_json(json_data, orient='split')
        
        return f"Showing {len(filtered_df)} accidents"
    
    @app.callback(
        Output('accident-map', 'figure'),
        [Input('filtered-data', 'data')]
    )
    def update_map(json_data):
        if not json_data:
            return {}
        
        # Parse the JSON data
        filtered_df = pd.read_json(json_data, orient='split')
        
        # Create map
        fig = go.Figure()
        
        # Add scattermapbox for accidents
        fig.add_trace(go.Scattermapbox(
            lat=filtered_df['latitude'],
            lon=filtered_df['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=filtered_df['severity'],
                colorscale='Viridis',
                colorbar=dict(title='Severity'),
                opacity=0.7
            ),
            text=filtered_df.apply(
                lambda row: f"Severity: {row['severity']}<br>Weather: {row['weather_condition']}<br>Road: {row['road_condition']}",
                axis=1
            ),
            hoverinfo='text'
        ))
        
        # Set map layout
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(
                    lat=filtered_df['latitude'].mean(),
                    lon=filtered_df['longitude'].mean()
                ),
                zoom=10
            ),
            margin=dict(r=0, t=0, l=0, b=0),
            height=600
        )
        
        return fig
    
    @app.callback(
        Output('severity-chart', 'figure'),
        [Input('filtered-data', 'data')]
    )
    def update_severity_chart(json_data):
        if not json_data:
            return {}
        
        # Parse the JSON data
        filtered_df = pd.read_json(json_data, orient='split')
        
        # Create severity count dataframe
        severity_counts = filtered_df['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        severity_counts = severity_counts.sort_values('severity')
        
        # Create chart
        fig = px.bar(
            severity_counts, 
            x='severity', 
            y='count',
            labels={'severity': 'Severity Level', 'count': 'Number of Accidents'},
            color='severity',
            color_continuous_scale='Viridis',
            text='count'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        fig.update_layout(
            xaxis=dict(tickmode='linear'),
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=30, b=40),
        )
        
        return fig
    
    @app.callback(
        Output('time-chart', 'figure'),
        [Input('filtered-data', 'data')]
    )
    def update_time_chart(json_data):
        if not json_data:
            return {}
        
        # Parse the JSON data
        filtered_df = pd.read_json(json_data, orient='split')
        
        # Create hour count dataframe
        hour_counts = filtered_df['hour'].value_counts().reset_index()
        hour_counts.columns = ['hour', 'count']
        hour_counts = hour_counts.sort_values('hour')
        
        # Create chart
        fig = px.line(
            hour_counts, 
            x='hour', 
            y='count',
            labels={'hour': 'Hour of Day', 'count': 'Number of Accidents'},
            markers=True
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', dtick=2),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        
        return fig
    
    @app.callback(
        Output('day-chart', 'figure'),
        [Input('filtered-data', 'data')]
    )
    def update_day_chart(json_data):
        if not json_data:
            return {}
        
        # Parse the JSON data
        filtered_df = pd.read_json(json_data, orient='split')
        
        # Create day count dataframe
        day_counts = filtered_df['day_name'].value_counts().reset_index()
        day_counts.columns = ['day', 'count']
        
        # Define the correct order of days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create a custom categorical type with our order
        day_counts['day'] = pd.Categorical(day_counts['day'], categories=days_order, ordered=True)
        
        # Sort by the ordered category
        day_counts = day_counts.sort_values('day')
        
        # Create chart
        fig = px.bar(
            day_counts, 
            x='day', 
            y='count',
            labels={'day': 'Day of Week', 'count': 'Number of Accidents'},
            color='count',
            color_continuous_scale='Viridis',
            text='count'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=30, b=40),
        )
        
        return fig
    
    @app.callback(
        Output('weather-chart', 'figure'),
        [Input('filtered-data', 'data')]
    )
    def update_weather_chart(json_data):
        if not json_data:
            return {}
        
        # Parse the JSON data
        filtered_df = pd.read_json(json_data, orient='split')
        
        # Create weather count dataframe
        weather_counts = filtered_df['weather_condition'].value_counts().reset_index()
        weather_counts.columns = ['weather', 'count']
        
        # Sort by count
        weather_counts = weather_counts.sort_values('count', ascending=False)
        
        # Create chart
        fig = px.pie(
            weather_counts, 
            names='weather', 
            values='count',
            title='Weather Conditions',
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        fig.update_layout(
            margin=dict(l=40, r=40, t=80, b=40),
        )
        
        return fig
    
    @app.callback(
        [Output('route-map-container', 'children'),
         Output('route-info', 'children')],
        [Input('generate-route-button', 'n_clicks')],
        [State('start-coords', 'value'),
         State('end-coords', 'value')]
    )
    def generate_safe_route(n_clicks, start_coords, end_coords):
        if n_clicks is None or n_clicks == 0:
            return [html.P("Enter start and end coordinates and click 'Generate Safe Route'")], []
        
        if not start_coords or not end_coords:
            return [html.P("Please enter both start and end coordinates")], []
        
        try:
            # Parse coordinates
            start = tuple(map(float, start_coords.split(',')))
            end = tuple(map(float, end_coords.split(',')))
            
            # Get hotspots (pre-computed)
            from analysis.hotspot_analysis import identify_hotspots_hdbscan
            hotspots = identify_hotspots_hdbscan()
            
            # Generate route
            from utils.routing import get_safe_routes
            route_map = get_safe_routes(start, end, hotspots)
            
            # Return iframe with map
            map_iframe = html.Iframe(
                srcDoc=open(route_map, 'r').read(),
                style={'width': '100%', 'height': '100%', 'border': 'none'}
            )
            
            route_info = [
                html.H4("Route Generated!"),
                html.P("The map shows two route options:"),
                html.Ul([
                    html.Li(html.Span("Standard Route (blue)", style={'color': 'blue'}), style={'marginBottom': '10px'}),
                    html.Li(html.Span("Safer Route (green)", style={'color': 'green'}))
                ]),
                html.P("The safer route avoids accident hotspots but may be slightly longer.")
            ]
            
            return [map_iframe], route_info
            
        except Exception as e:
            return [html.P(f"Error generating route: {str(e)}")], []
    
    # Add route for serving static files from output directory
    @app.server.route('/output/<path:path>')
    def serve_static(path):
        from flask import send_from_directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return send_from_directory(os.path.join(root_dir, 'output'), path)
    
    print("Dashboard initialized and ready to run")
    return app

def run_dashboard(debug=True, port=8090):
    """Run the dashboard server"""
    app = create_interactive_dashboard()
    
    # Display the URL to access the dashboard
    print(f"\nDashboard is running at http://localhost:{port}/")
    print("Press Ctrl+C to stop the server\n")
    
    # Start the server using the current method
    app.run(debug=debug, port=port)

if __name__ == "__main__":
    print("\n=== RUNNING INTERACTIVE DASHBOARD ===\n")
    run_dashboard()
