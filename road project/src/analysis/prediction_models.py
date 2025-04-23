import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from joblib import dump, load
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database import get_db_connection

# Create a directory for models
os.makedirs('models', exist_ok=True)

def build_severity_prediction_model():
    """
    Build and evaluate a model to predict accident severity
    
    Returns:
        Trained model and evaluation metrics
    """
    print("Building accident severity prediction model...")
    
    # Get database connection
    conn = get_db_connection()
    
    # Get accident data with features
    query = """
    SELECT 
        severity,
        strftime('%H', timestamp) as hour,
        strftime('%w', timestamp) as day_of_week,
        weather_condition,
        road_condition,
        light_condition,
        vehicles_involved,
        latitude,
        longitude
    FROM accidents
    WHERE severity IS NOT NULL
    """
    df = pd.read_sql(query, conn)
        # Add a few simple features that might help prediction
    # Create new features
    df['is_night'] = (df['hour'].astype(int) >= 20) | (df['hour'].astype(int) <= 5)
    df['is_rush_hour'] = df['hour'].astype(int).isin([7, 8, 9, 16, 17, 18])
    df['is_weekend'] = df['day_of_week'].astype(int).isin([0, 6])  # 0=Sunday, 6=Saturday
    
    # Bad conditions indicators
    df['bad_weather'] = df['weather_condition'].isin(['Rain', 'Snow', 'Fog'])
    df['bad_road'] = df['road_condition'].isin(['Wet', 'Snow', 'Ice'])

    # Feature engineering
    # One-hot encode categorical variables
    categorical_cols = ['weather_condition', 'road_condition', 'light_condition']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Convert hour to cyclical features to represent time of day
    df_encoded['hour_sin'] = np.sin(2 * np.pi * df_encoded['hour'].astype(float)/24)
    df_encoded['hour_cos'] = np.cos(2 * np.pi * df_encoded['hour'].astype(float)/24)
    
    # Convert day of week to cyclical features
    df_encoded['day_sin'] = np.sin(2 * np.pi * df_encoded['day_of_week'].astype(float)/7)
    df_encoded['day_cos'] = np.cos(2 * np.pi * df_encoded['day_of_week'].astype(float)/7)
    
    # Drop original hour and day_of_week columns
    df_encoded.drop(['hour', 'day_of_week'], axis=1, inplace=True)
    
    # Define X and y
    X = df_encoded.drop('severity', axis=1)
    y = df_encoded['severity']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature importance
    importances = model.feature_importances_
    feature_importance = dict(zip(X.columns, importances))
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    # Save model and scaler
    dump(model, 'models/severity_prediction_model.joblib')
    dump(scaler, 'models/severity_prediction_scaler.joblib')
    
    # Close connection
    conn.close()
    
    print(f"Severity prediction model accuracy: {accuracy:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': feature_importance,
        'features': list(X.columns)
    }

def get_possible_values(column):
    """Get all possible values for a categorical column"""
    # This would be better implemented by querying the database
    # for all unique values, but for simplicity:
    possible_values = {
        'road_type': ['primary', 'secondary', 'tertiary', 'residential'],
        'surface': ['asphalt', 'concrete'],
        'weather_condition': ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Unknown'],
        'road_condition': ['Dry', 'Wet', 'Snow', 'Ice', 'Slush', 'Unknown'],
        'light_condition': ['Daylight', 'Dark - lights on', 'Dark - no lights', 'Dawn/Dusk']
    }
    return possible_values.get(column, [])

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 0.2:
        return "Low"
    elif probability < 0.4:
        return "Moderate"
    elif probability < 0.6:
        return "Elevated"
    elif probability < 0.8:
        return "High"
    else:
        return "Severe"
        
def build_accident_probability_model():
    """
    Build and evaluate a model to predict accident probability on road segments
    
    Returns:
        Trained model and evaluation metrics
    """
    print("Building accident probability prediction model...")
    
    try:
        # Get database connection
        conn = get_db_connection()
        
        # First, get all road segments
        road_segments_df = pd.read_sql("SELECT * FROM road_segments", conn)
        
        # Next, get all accidents
        accidents_df = pd.read_sql("""
            SELECT 
                accident_id, latitude, longitude, severity, timestamp,
                weather_condition, road_condition, light_condition
            FROM accidents
        """, conn)
        
        # Convert timestamp to datetime
        accidents_df['timestamp'] = pd.to_datetime(accidents_df['timestamp'])
        
        # Extract time features
        accidents_df['hour'] = accidents_df['timestamp'].dt.hour
        accidents_df['day_of_week'] = accidents_df['timestamp'].dt.dayofweek
        
        # Associate each accident with the nearest road segment
        print("Matching accidents to road segments...")
        
        def find_nearest_road_segment(lat, lon):
            """Find the nearest road segment to a given point"""
            min_distance = float('inf')
            nearest_segment = None
            
            for _, segment in road_segments_df.iterrows():
                # Calculate distance to line segment
                try:
                    start = (float(segment['start_lat']), float(segment['start_lon']))
                    end = (float(segment['end_lat']), float(segment['end_lon']))
                    
                    # Use projection to find nearest point on line segment
                    # (This is a simplified approach - more complex algorithms exist)
                    p = (lat, lon)
                    
                    # Vector projection calculation
                    v = (end[0] - start[0], end[1] - start[1])
                    w = (p[0] - start[0], p[1] - start[1])
                    c1 = sum(x*y for x, y in zip(w, v))
                    c2 = sum(x*x for x in v)
                    
                    if c2 == 0:
                        # Line segment is a point
                        point_on_segment = start
                    else:
                        b = c1 / c2
                        if b < 0:
                            point_on_segment = start
                        elif b > 1:
                            point_on_segment = end
                        else:
                            point_on_segment = (start[0] + b * v[0], start[1] + b * v[1])
                    
                    # Calculate distance to nearest point on segment
                    distance = geodesic(p, point_on_segment).km
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_segment = segment['segment_id']
                except (ValueError, TypeError) as e:
                    # Skip segments with invalid coordinates
                    continue
            
            return nearest_segment, min_distance
        
        # This is computationally intensive - apply to a subset for demonstration
        sample_size = min(1000, len(accidents_df))
        accidents_sample = accidents_df.sample(sample_size, random_state=42)
        
        # Map accidents to road segments
        segment_mappings = []
        for _, accident in accidents_sample.iterrows():
            segment_id, distance = find_nearest_road_segment(
                accident['latitude'], accident['longitude']
            )
            
            if segment_id is not None and distance <= 0.1:  # Only consider accidents within 100m of a road
                segment_mappings.append({
                    'accident_id': accident['accident_id'],
                    'segment_id': segment_id,
                    'distance': distance
                })
        
        if len(segment_mappings) < 10:
            conn.close()
            raise ValueError("Not enough accidents could be matched to road segments. Need at least 10 matches.")
            
        mapping_df = pd.DataFrame(segment_mappings)
        
        # Merge mapping with accident details
        accident_segments = pd.merge(
            mapping_df, accidents_sample, on='accident_id'
        )
        
        # Create training dataset
        # For each road segment, gather occurrences of accidents and non-accidents
        
        # First, get accident counts per segment
        segment_accident_counts = accident_segments['segment_id'].value_counts()
        
        # Create positive samples (accidents)
        positive_samples = []
        for _, row in accident_segments.iterrows():
            positive_samples.append({
                'segment_id': row['segment_id'],
                'hour': row['hour'],
                'day_of_week': row['day_of_week'],
                'weather_condition': row['weather_condition'],
                'road_condition': row['road_condition'],
                'light_condition': row['light_condition'],
                'has_accident': 1
            })
        
        # Create negative samples (no accidents)
        # For each segment with accidents, create "safe" timepoints
        negative_samples = []
        
        for segment_id in segment_accident_counts.index:
            accident_times = accident_segments[
                accident_segments['segment_id'] == segment_id
            ]['timestamp'].tolist()
            
            # Get road segment details
            segment = road_segments_df[road_segments_df['segment_id'] == segment_id].iloc[0]
            
            # Generate random non-accident times for this segment
            # (simplified approach - in reality, would need more sophisticated sampling)
            for _ in range(min(10, len(accident_times) * 2)):  # 2x negative samples
                # Random timestamp within data range
                random_hour = np.random.randint(0, 24)
                random_day = np.random.randint(0, 7)
                
                negative_samples.append({
                    'segment_id': segment_id,
                    'hour': random_hour,
                    'day_of_week': random_day,
                    'weather_condition': np.random.choice(['Clear', 'Rain', 'Snow']),
                    'road_condition': np.random.choice(['Dry', 'Wet', 'Snow']),
                    'light_condition': 'Daylight' if 6 <= random_hour <= 18 else 'Dark - lights on',
                    'has_accident': 0
                })
        
        # Combine positive and negative samples
        training_data = pd.DataFrame(positive_samples + negative_samples)
        
        # Get only numeric features from road segments
        numeric_features = ['start_lat', 'start_lon', 'end_lat', 'end_lon', 
                            'lanes', 'speed_limit', 'oneway']
        
        # Merge with road segment details (only numeric features)
        road_segments_subset = road_segments_df[['segment_id'] + numeric_features]
        training_data = pd.merge(
            training_data, 
            road_segments_subset,
            on='segment_id'
        )
        
        # Feature engineering
        # One-hot encode categorical variables
        categorical_cols = ['weather_condition', 'road_condition', 'light_condition']
        training_encoded = pd.get_dummies(training_data, columns=categorical_cols)
        
        # Convert hour to cyclical features
        training_encoded['hour_sin'] = np.sin(2 * np.pi * training_encoded['hour']/24)
        training_encoded['hour_cos'] = np.cos(2 * np.pi * training_encoded['hour']/24)
        
        # Convert day of week to cyclical features
        training_encoded['day_sin'] = np.sin(2 * np.pi * training_encoded['day_of_week']/7)
        training_encoded['day_cos'] = np.cos(2 * np.pi * training_encoded['day_of_week']/7)
        
        # Drop unnecessary columns
        cols_to_drop = ['hour', 'day_of_week', 'segment_id']
        training_encoded.drop(cols_to_drop, axis=1, inplace=True)
        
        # Define X and y
        X = training_encoded.drop('has_accident', axis=1)
        y = training_encoded['has_accident']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (using Random Forest for simplicity)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save models
        dump(model, 'models/accident_probability_model.joblib')
        dump(scaler, 'models/accident_probability_scaler.joblib')
        
        # Save feature names for later reference
        with open('models/accident_probability_features.json', 'w') as f:
            json.dump(list(X.columns), f)
        
        # Close connection
        conn.close()
        
        print(f"Accident probability model - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'auc': auc,
            'report': report,
            'features': list(X.columns)
        }
    
    except Exception as e:
        print(f"Warning: Accident probability model failed: {e}")
        print("This might be due to insufficient data for road-accident association.")
        return None


def predict_accident_probability(road_segment_id=None, conditions=None):
    """
    Predict the probability of accidents occurring on a specific road segment
    under given conditions
    
    Args:
        road_segment_id: ID of the road segment to analyze
        conditions: Dictionary of conditions (weather, time, etc.)
    
    Returns:
        Predicted probability and risk factors
    """
    print(f"Predicting accident probability for road segment {road_segment_id}...")
    
    # Get database connection
    conn = get_db_connection()
    
    # If no conditions provided, use current conditions
    if conditions is None:
        conditions = {
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'weather_condition': 'Clear',  # Default weather
            'road_condition': 'Dry',       # Default road condition
            'light_condition': 'Daylight'  # Default light condition
        }
    
    # Get road segment data
    cursor = conn.cursor()
    cursor.execute("""
        SELECT segment_id, start_lat, start_lon, end_lat, end_lon, 
               road_type, lanes, speed_limit, oneway, surface
        FROM road_segments
        WHERE segment_id = ?
    """, (road_segment_id,))
    road = cursor.fetchone()
    
    if road is None:
        conn.close()
        return {'error': f'Road segment {road_segment_id} not found'}
    
    # Try to load the model, but if it fails, return a mock prediction
    # This allows the visualization to work without requiring model training
    model_path = 'models/accident_probability_model.joblib'
    scaler_path = 'models/accident_probability_scaler.joblib'
    features_path = 'models/accident_probability_features.json'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        try:
            model = load(model_path)
            scaler = load(scaler_path)
            
            # Load feature names 
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
            
            # Prepare input features (only numeric features)
            features = {
                'start_lat': float(road['start_lat']),
                'start_lon': float(road['start_lon']),
                'end_lat': float(road['end_lat']),
                'end_lon': float(road['end_lon']),
                'lanes': int(road['lanes']),
                'speed_limit': int(road['speed_limit']),
                'oneway': int(road['oneway']),
                'hour_sin': np.sin(2 * np.pi * conditions['hour']/24),
                'hour_cos': np.cos(2 * np.pi * conditions['hour']/24),
                'day_sin': np.sin(2 * np.pi * conditions['day_of_week']/7),
                'day_cos': np.cos(2 * np.pi * conditions['day_of_week']/7),
            }
            
            # Add one-hot encoded features for categorical variables
            for col in ['weather_condition', 'road_condition', 'light_condition']:
                value = conditions[col]
                
                # Create one-hot encoding
                for possible_value in get_possible_values(col):
                    feature_name = f"{col}_{possible_value}"
                    features[feature_name] = 1 if value == possible_value else 0
            
            # Convert to DataFrame
            input_df = pd.DataFrame([features])
            
            # Ensure all expected columns are present and in the right order
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Select only the columns the model knows about
            input_df = input_df[feature_names]
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_scaled)[0][1]  # Probability of class 1 (accident)
            else:
                # For models that don't support predict_proba
                prediction = model.predict(input_scaled)[0]
                probability = prediction
            
            # Get feature importance for this prediction
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_names, importances))
                top_factors = {k: v for k, v in sorted(feature_importance.items(), 
                                                     key=lambda item: item[1] * abs(input_df[item[0]].values[0]), 
                                                     reverse=True)[:5]}
            else:
                top_factors = {}
                
            conn.close()
            
            return {
                'road_segment_id': road_segment_id,
                'accident_probability': float(probability),
                'risk_level': get_risk_level(probability),
                'top_risk_factors': top_factors,
                'conditions': conditions
            }
            
        except Exception as e:
            print(f"Error using trained model: {e}")
            print("Using mock prediction instead.")
            # Continue to mock prediction below
    else:
        print("Model not found. Using mock prediction instead.")
    
    # Rest of the function (mock prediction) remains the same...

    
    # If the model loading or prediction failed, use a deterministic mock prediction
    # This ensures the visualization works even without a trained model
    
    # Generate a deterministic but varied probability based on road properties
    # This ensures the visualization shows varied risk levels
    road_type_risk = {
        'primary': 0.7,
        'secondary': 0.5,
        'tertiary': 0.3,
        'residential': 0.2
    }
    
    # Base probability on road type and other features
    base_prob = road_type_risk.get(road['road_type'], 0.4)
    
    # Adjust based on speed limit (higher speed = higher risk)
    speed_factor = min(1.5, road['speed_limit'] / 40)
    
    # Adjust based on number of lanes (more lanes = higher risk)
    lane_factor = 0.8 + (0.1 * road['lanes'])
    
    # Calculate final probability
    mock_probability = base_prob * speed_factor * lane_factor
    
    # Ensure it's in the valid range
    mock_probability = max(0.1, min(0.9, mock_probability))
    
    # Create some mock risk factors
    mock_top_factors = {
        f"road_type_{road['road_type']}": 0.4,
        "speed_limit": 0.3,
        f"lanes_{road['lanes']}": 0.2,
        f"weather_condition_{conditions['weather_condition']}": 0.15,
        f"road_condition_{conditions['road_condition']}": 0.1
    }
    
    conn.close()
    
    return {
        'road_segment_id': road_segment_id,
        'accident_probability': float(mock_probability),
        'risk_level': get_risk_level(mock_probability),
        'top_risk_factors': mock_top_factors,
        'conditions': conditions,
        'is_mock': True  # Indicate this is a mock prediction
    }


if __name__ == "__main__":
    print("\n=== PREDICTIVE MODEL TRAINING ===\n")
    
    # Train severity prediction model
    severity_model = build_severity_prediction_model()
    print(f"Severity model accuracy: {severity_model['accuracy']:.4f}")
    print("Top 5 features for severity:")
    top_features = list(severity_model['feature_importance'].items())[:5]
    for feature, importance in top_features:
        print(f"  - {feature}: {importance:.4f}")
    
    # Train accident probability model
    probability_model = build_accident_probability_model()
    print(f"Accident probability model - Accuracy: {probability_model['accuracy']:.4f}, AUC: {probability_model['auc']:.4f}")
    
    # Test prediction on a sample road segment
    sample_prediction = predict_accident_probability('RD-EW-1')
    print("\nSample prediction:")
    print(f"Road segment: {sample_prediction['road_segment_id']}")