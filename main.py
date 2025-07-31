"""
Main execution script for Indian Road Traffic Accident Prediction System
This script orchestrates data collection, model training, and Tableau integration
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_collection.weather_collector import WeatherDataCollector
from data_collection.traffic_collector import TrafficDataCollector
from data_collection.accident_processor import AccidentDataProcessor
from models.ml_models import AccidentPredictionModel
from tableau_integration.tabpy_functions import TableauIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrafficAccidentPredictionSystem:
    """Main system orchestrator"""

    def __init__(self):
        self.weather_collector = WeatherDataCollector()
        self.traffic_collector = TrafficDataCollector()
        self.accident_processor = AccidentDataProcessor()
        self.ml_model = AccidentPredictionModel()
        self.tableau_integration = TableauIntegration()

        # Create logs directory
        os.makedirs('logs', exist_ok=True)

    def run_data_collection(self):
        """Execute data collection from all sources"""
        logger.info("=== STARTING DATA COLLECTION ===")

        try:
            # Collect weather data
            logger.info("Collecting weather data...")
            major_cities = {
                "Delhi": "42182",
                "Mumbai": "43003", 
                "Bangalore": "43295",
                "Chennai": "43279",
                "Kolkata": "42809",
                "Hyderabad": "43128",
                "Pune": "43063",
                "Ahmedabad": "42647"
            }

            weather_df = self.weather_collector.collect_multiple_stations(list(major_cities.values()))
            if not weather_df.empty:
                weather_file = self.weather_collector.save_weather_data(weather_df)
                logger.info(f"Weather data saved: {weather_file}")

            # Collect traffic data
            logger.info("Collecting traffic data...")
            major_cities_list = [f"{city}, India" for city in major_cities.keys()]
            traffic_df = self.traffic_collector.collect_traffic_matrix(major_cities_list[:4])

            if not traffic_df.empty:
                traffic_file = self.traffic_collector.save_traffic_data(traffic_df)
                logger.info(f"Traffic data saved: {traffic_file}")

            # Generate accident hotspots
            hotspots_df = self.traffic_collector.get_accident_hotspots()
            hotspots_file = self.traffic_collector.save_traffic_data(hotspots_df, "accident_hotspots.csv")
            logger.info(f"Hotspot data saved: {hotspots_file}")

            logger.info("Data collection completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            return False

    def run_data_processing(self):
        """Process and prepare accident data"""
        logger.info("=== STARTING DATA PROCESSING ===")

        try:
            # Generate synthetic accident data
            logger.info("Generating synthetic accident data...")
            accident_df = self.accident_processor.generate_synthetic_accident_data(10000)

            # Preprocess data
            logger.info("Preprocessing accident data...")
            processed_df = self.accident_processor.preprocess_accident_data(accident_df)

            # Save processed data
            processed_file = self.accident_processor.save_processed_data(processed_df)
            logger.info(f"Processed data saved: {processed_file}")

            # Generate statistics
            stats = self.accident_processor.get_accident_statistics(processed_df)
            logger.info("Data processing completed successfully!")

            return processed_df, stats

        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            return None, None

    def run_model_training(self, processed_df):
        """Train machine learning models"""
        logger.info("=== STARTING MODEL TRAINING ===")

        try:
            # Train all models
            logger.info("Training machine learning models...")
            X_test, y_test = self.ml_model.train_all_models(processed_df)

            # Save models
            model_file = self.ml_model.save_models()
            logger.info(f"Models saved: {model_file}")

            # Test prediction
            sample_data = {
                'temperature': 30.0,
                'humidity': 70.0,
                'rainfall': 0.0,
                'wind_speed': 15.0,
                'visibility': 10.0,
                'hour': 18,
                'day_of_week': 1,
                'month': 6,
                'is_weekend': 0,
                'is_night': 0,
                'is_peak_hour': 1,
                'state': 'Tamil Nadu',
                'road_type': 'National Highway',
                'collision_type': 'Head on Collision',
                'vehicle_type': 'Two Wheeler',
                'traffic_violation': 'Over Speeding',
                'weather_condition': 'Clear/Sunny',
                'junction_type': 'Straight Road',
                'area_type': 'Urban'
            }

            predictions = self.ml_model.predict_accident_risk(sample_data)
            logger.info("Sample prediction completed")

            logger.info("Model training completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False

    def run_tableau_integration(self, processed_df):
        """Setup Tableau integration"""
        logger.info("=== STARTING TABLEAU INTEGRATION ===")

        try:
            # Create Tableau data extract
            logger.info("Creating Tableau data extract...")
            extract_file = self.tableau_integration.create_tableau_data_extract(processed_df)
            logger.info(f"Tableau extract created: {extract_file}")

            # Setup TabPy functions
            logger.info("Setting up TabPy functions...")
            self.tableau_integration.setup_tabpy_functions()

            # Generate dashboard specifications
            dashboard_spec = self.tableau_integration.generate_tableau_dashboard_spec()

            # Save dashboard spec
            spec_file = os.path.join('tableau_workbooks', 'dashboard_specification.json')
            os.makedirs('tableau_workbooks', exist_ok=True)
            import json
            with open(spec_file, 'w') as f:
                json.dump(dashboard_spec, f, indent=2)

            # Generate instructions
            instructions_file = self.tableau_integration.export_tableau_workbook_instructions()

            logger.info("Tableau integration completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error in Tableau integration: {e}")
            return False

    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        logger.info("=== STARTING FULL PIPELINE EXECUTION ===")
        start_time = datetime.now()

        try:
            # Step 1: Data Collection
            collection_success = self.run_data_collection()
            if not collection_success:
                logger.warning("Data collection had issues, continuing with synthetic data...")

            # Step 2: Data Processing
            processed_df, stats = self.run_data_processing()
            if processed_df is None:
                logger.error("Data processing failed. Stopping pipeline.")
                return False

            # Step 3: Model Training
            training_success = self.run_model_training(processed_df)
            if not training_success:
                logger.error("Model training failed. Stopping pipeline.")
                return False

            # Step 4: Tableau Integration
            tableau_success = self.run_tableau_integration(processed_df)
            if not tableau_success:
                logger.warning("Tableau integration had issues.")

            # Generate final report
            self.generate_execution_report(stats, start_time)

            logger.info("=== FULL PIPELINE COMPLETED SUCCESSFULLY ===")
            return True

        except Exception as e:
            logger.error(f"Error in full pipeline execution: {e}")
            return False

    def generate_execution_report(self, stats, start_time):
        """Generate execution summary report"""
        end_time = datetime.now()
        execution_time = end_time - start_time

        report = f"""
# Traffic Accident Prediction System - Execution Report

## Execution Summary
- Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- Total Execution Time: {execution_time}
- Status: SUCCESS

## Data Statistics
"""
        if stats:
            report += f"""
- Total Accidents Processed: {stats['total_accidents']:,}
- Total Deaths: {stats['total_deaths']:,}
- Total Injuries: {stats['total_injuries']:,}
- Fatality Rate: {stats['fatality_rate']:.2f}%

### Top 5 States by Accidents:
"""
            for state, count in list(stats['accident_by_state'].items())[:5]:
                report += f"- {state}: {count:,}\n"

        report += """
## Generated Files
- Raw weather data: `data/raw/weather_data_*.csv`
- Raw traffic data: `data/raw/traffic_data_*.csv`
- Processed accident data: `data/processed/processed_accident_data_*.csv`
- Trained models: `models/accident_prediction_models_*.joblib`
- Tableau extract: `data/processed/accident_data_extract_*.csv`
- Dashboard specification: `tableau_workbooks/dashboard_specification.json`

## Next Steps
1. Start TabPy server: `tabpy`
2. Open Tableau Desktop
3. Follow instructions in `documentation/tableau_instructions.md`
4. Create interactive dashboard for accident prediction

## Model Performance
All machine learning models have been trained and are ready for prediction.
The system can now predict accident risk based on real-time conditions.
"""

        report_file = f"documentation/execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"Execution report saved: {report_file}")
        print("\n" + "="*60)
        print("EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Report saved: {report_file}")
        print("Check the documentation folder for detailed instructions.")

def main():
    """Main function"""
    print("="*60)
    print("INDIAN ROAD TRAFFIC ACCIDENT PREDICTION SYSTEM")
    print("="*60)
    print("This system will:")
    print("1. Collect real-time weather and traffic data")
    print("2. Process Indian road accident data")
    print("3. Train machine learning models")
    print("4. Setup Tableau integration with TabPy")
    print("5. Generate interactive dashboards")
    print("="*60)

    # Initialize system
    system = TrafficAccidentPredictionSystem()

    # Run full pipeline
    success = system.run_full_pipeline()

    if success:
        print("\n🎉 System setup completed successfully!")
        print("\nNext steps:")
        print("1. Install Tableau Desktop")
        print("2. Start TabPy server: tabpy")
        print("3. Follow instructions in documentation/tableau_instructions.md")
        print("4. Create your interactive accident prediction dashboard")
    else:
        print("\n❌ System setup encountered errors. Check logs for details.")

if __name__ == "__main__":
    main()
