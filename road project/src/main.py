import os
import sys
import argparse
from datetime import datetime

def main():
    """Main entry point for the road safety analysis system"""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Road Safety Analysis System')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--import-real-data', action='store_true', help='Import real road network with synthetic accidents')
    parser.add_argument('--city', type=str, default='Chicago', help='City to import data for')
    parser.add_argument('--country', type=str, default='Illinois', help='Country/State of the city')
    parser.add_argument('--analysis', action='store_true', help='Run hotspot analysis')
    parser.add_argument('--train-models', action='store_true', help='Train prediction models')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--dashboard', action='store_true', help='Launch interactive dashboard')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--max-roads', type=int, default=500, help='Maximum number of road segments to import')
    parser.add_argument('--num-accidents', type=int, default=500, help='Number of synthetic accidents to generate')
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run all steps if --all is specified
    if args.all:
        args.analysis = True
        args.train_models = True  
        args.visualize = True
        args.dashboard = True
        # Note: We don't auto-set generate-data or import-real-data in --all mode
        # to avoid overwriting existing data
    
    start_time = datetime.now()
    print(f"\n=== ROAD SAFETY ANALYSIS SYSTEM ===")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Data source handling - check what data is available
    db_file = 'data/db/road_safety.db'
    has_data = False
    
    if os.path.exists(db_file):
        try:
            # Connect to the database
            import sqlite3
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Check if there are road segments and accidents
            cursor.execute("SELECT COUNT(*) FROM road_segments")
            road_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM accidents")
            accident_count = cursor.fetchone()[0]
            
            conn.close()
            
            # If we have a reasonable amount of data, consider it valid
            if road_count > 0 and accident_count > 0:
                print(f"=== USING EXISTING DATA: {road_count} road segments and {accident_count} accidents ===")
                has_data = True
        except Exception as e:
            print(f"Error checking database: {e}")
            has_data = False
    
    # Step 1: Generate synthetic data OR import real data if needed
    if args.generate_data and not (has_data and args.import_real_data):
        print("\n=== GENERATING SYNTHETIC DATA ===\n")
        from utils.data_generator import generate_accident_data, generate_road_network_data
        from utils.database import setup_database, load_sample_data
        
        # Setup database
        setup_database()
        
        # Generate data
        accident_data = generate_accident_data(2000)
        road_data = generate_road_network_data()
        
        # Load generated data into database
        load_sample_data()
        
        has_data = True
    
    # Step 1b: Import real data if requested
    # Add these parameters to your ArgumentParser

    # In the import_real_data section:
    if args.import_real_data and not has_data:
        print(f"\n=== IMPORTING REAL ROAD NETWORK FOR {args.city}, {args.country} ===\n")
        try:
            from utils.database import setup_database
            setup_database()
            
            from utils.real_data_importer import import_real_data
            
            # Import real data with limits
            import_real_data(args.city, args.country, args.max_roads, args.num_accidents)
            has_data = True
        except ImportError as e:
            print(f"Error importing real data modules: {e}")
            print("\nMake sure you have installed the required packages:")
            print("pip install osmnx geopandas shapely geopy tqdm")
        except Exception as e:
            print(f"Error importing real data: {e}")

    # Exit if we still don't have data
    if not has_data:
        print("\nERROR: No data available. Please generate data or import real data first.")
        print("Run with --generate-data or --import-real-data")
        return
    
    # Step 2: Run hotspot analysis
    if args.analysis:
        print("\n=== RUNNING HOTSPOT ANALYSIS ===\n")
        from analysis.hotspot_analysis import (
            identify_hotspots_simple, 
            identify_hotspots_dbscan,
            identify_hotspots_hdbscan,
            identify_spatiotemporal_hotspots,
            calculate_risk_factors
        )
        
        print("Performing basic hotspot analysis...")
        simple_hotspots = identify_hotspots_simple()
        dbscan_hotspots = identify_hotspots_dbscan()
        
        print("\nPerforming advanced hotspot analysis...")
        try:
            hdbscan_hotspots = identify_hotspots_hdbscan()
            print(f"HDBSCAN identified {len(hdbscan_hotspots)} hotspots")
        except Exception as e:
            print(f"Warning: HDBSCAN analysis failed: {e}")
            print("Install hdbscan package with: pip install hdbscan")
            hdbscan_hotspots = []
        
        try:
            spatiotemporal_hotspots = identify_spatiotemporal_hotspots()
            print(f"Spatio-temporal analysis identified {len(spatiotemporal_hotspots)} hotspots")
        except Exception as e:
            print(f"Warning: Spatio-temporal analysis failed: {e}")
            spatiotemporal_hotspots = []
        
        # Risk factor analysis
        print("\nCalculating risk factors...")
        risk_factors = calculate_risk_factors()
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total accidents analyzed: {risk_factors['total_accidents']}")
        print(f"Hotspots identified (simple method): {len(simple_hotspots)}")
        print(f"Hotspots identified (DBSCAN): {len(dbscan_hotspots)}")
        print(f"Hotspots identified (HDBSCAN): {len(hdbscan_hotspots)}")
        print(f"Spatio-temporal hotspots identified: {len(spatiotemporal_hotspots)}")
        
        # Print top hotspots
        if hdbscan_hotspots:
            print("\nTop 5 hotspots (HDBSCAN):")
            for i, hotspot in enumerate(hdbscan_hotspots[:5], 1):
                print(f"  {i}. {hotspot['accident_count']} accidents, avg severity: {hotspot['avg_severity']:.2f}")
                if 'risk_score' in hotspot:
                    print(f"     Risk score: {hotspot['risk_score']:.2f}")
    
    # Step 3: Train prediction models
    if args.train_models:
        print("\n=== TRAINING PREDICTION MODELS ===\n")
        try:
            from analysis.prediction_models import (
                build_severity_prediction_model,
                build_accident_probability_model,
                predict_accident_probability
            )
            
            print("Training severity prediction model...")
            severity_model = build_severity_prediction_model()
            print(f"Severity model accuracy: {severity_model['accuracy']:.4f}")
            print("Top 5 features for severity:")
            top_features = list(severity_model['feature_importance'].items())[:5]
            for feature, importance in top_features:
                print(f"  - {feature}: {importance:.4f}")
            
            print("\nTraining accident probability model...")
            try:
                accident_probability_model = build_accident_probability_model()
                print(f"Accident probability model - Accuracy: {accident_probability_model['accuracy']:.4f}")
                if 'auc' in accident_probability_model:
                    print(f"AUC: {accident_probability_model['auc']:.4f}")
                
                # Test prediction on a sample road segment
                print("\nTesting prediction on sample road segment...")
                # Get a random segment ID from the database
                import sqlite3
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT segment_id FROM road_segments LIMIT 1")
                segment_id = cursor.fetchone()[0]
                conn.close()
                
                sample_prediction = predict_accident_probability(segment_id)
                print(f"Road segment: {sample_prediction['road_segment_id']}")
                print(f"Accident probability: {sample_prediction['accident_probability']:.4f}")
                print(f"Risk level: {sample_prediction['risk_level']}")
                
                if 'top_risk_factors' in sample_prediction:
                    print("Top risk factors:")
                    for factor, value in list(sample_prediction['top_risk_factors'].items())[:3]:
                        print(f"  - {factor}: {value:.4f}")
            except Exception as e:
                print(f"Warning: Accident probability model failed: {e}")
                print("This might be due to insufficient data for road-accident association.")
                
        except ImportError as e:
            print(f"Error: Could not import prediction modules: {e}")
            print("Check that you've created the prediction_models.py file and installed required packages.")
    
    # Step 4: Generate visualizations
    if args.visualize:
        print("\n=== GENERATING VISUALIZATIONS ===\n")
        try:
            from visualization.map_generator import (
                create_accident_map,
                create_road_segment_map, 
                create_heatmap,
                create_risk_factor_visualizations,
                create_severity_distribution_chart,
                create_time_series_analysis
            )
            
            # Ensure output directories exist
            os.makedirs('output/maps', exist_ok=True)
            os.makedirs('output/charts', exist_ok=True)
            
            # Get hotspots for visualization (if not already calculated)
            if not args.analysis:
                from analysis.hotspot_analysis import (
                    identify_hotspots_dbscan,
                    calculate_risk_factors
                )
                
                print("Getting hotspots for visualization...")
                dbscan_hotspots = identify_hotspots_dbscan()
                risk_factors = calculate_risk_factors()
                
                try:
                    from analysis.hotspot_analysis import identify_hotspots_hdbscan
                    hdbscan_hotspots = identify_hotspots_hdbscan()
                except Exception:
                    hdbscan_hotspots = []
            
            # Create basic visualizations
            print("Creating basic visualizations...")
            create_accident_map(dbscan_hotspots)
            create_road_segment_map()
            create_heatmap()
            create_risk_factor_visualizations(risk_factors)
            create_severity_distribution_chart()
            create_time_series_analysis()
            
            # Create enhanced visualizations if functions are defined
            print("\nCreating enhanced visualizations...")
            
            # Check if hdbscan hotspots are available
            if 'hdbscan_hotspots' in locals() and hdbscan_hotspots:
                try:
                    # Import the function if it exists
                    from visualization.map_generator import create_hdbscan_hotspot_map
                    create_hdbscan_hotspot_map(hdbscan_hotspots)
                except ImportError:
                    print("Warning: HDBSCAN hotspot map function not found.")
            
            # Create predictive risk map if function exists
            try:
                from visualization.map_generator import create_predictive_risk_map
                create_predictive_risk_map()
            except ImportError:
                print("Warning: Predictive risk map function not defined.")
            except Exception as e:
                print(f"Warning: Could not create predictive risk map: {e}")
                print("Run with --train-models first to train the prediction models")
            
            print("\nAll visualizations created successfully!")
            
        except ImportError as e:
            print(f"Error: Could not import visualization modules: {e}")
            print("Check that visualization modules are properly implemented.")
    
    # Step 5: Launch interactive dashboard
    if args.dashboard:
        print("\n=== LAUNCHING INTERACTIVE DASHBOARD ===\n")
        try:
            from dashboard.app import run_dashboard
            run_dashboard(debug=True)
        except ImportError as e:
            print(f"Error: Could not import dashboard module: {e}")
            print("Check that dashboard module is properly implemented and required packages are installed.")
            print("Required packages: dash, plotly")
    
    # Calculate and display execution time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n=== EXECUTION COMPLETE ===")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main()
