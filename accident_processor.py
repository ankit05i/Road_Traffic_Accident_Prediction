"""
Accident Data Processor for Indian Traffic Accident Prediction
Processes official Indian road accident data from MORTH reports
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_SOURCES, FEATURE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccidentDataProcessor:
    """Processes Indian road accident data from official sources"""

    def __init__(self):
        self.raw_data_path = 'data/raw/'
        self.processed_data_path = 'data/processed/'

    def generate_synthetic_accident_data(self, num_records: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic accident data based on Indian road accident patterns
        This simulates the structure of actual MORTH accident data
        """
        np.random.seed(42)
        random.seed(42)

        # Indian states with their accident frequencies (based on MORTH 2022 data)
        states_data = {
            'Tamil Nadu': 0.139,
            'Madhya Pradesh': 0.118,
            'Kerala': 0.095,
            'Uttar Pradesh': 0.090,
            'Karnataka': 0.086,
            'Maharashtra': 0.072,
            'Rajasthan': 0.051,
            'Telangana': 0.047,
            'Andhra Pradesh': 0.046,
            'Gujarat': 0.034,
            'West Bengal': 0.028,
            'Haryana': 0.025,
            'Bihar': 0.023,
            'Odisha': 0.020,
            'Punjab': 0.018,
            'Others': 0.128
        }

        # Generate synthetic data
        data = []

        for i in range(num_records):
            # Select state based on probability
            state = np.random.choice(list(states_data.keys()), p=list(states_data.values()))

            # Generate timestamp (last 3 years)
            start_date = datetime(2021, 1, 1)
            end_date = datetime(2024, 12, 31)
            time_between = end_date - start_date
            days_between = time_between.days
            random_days = random.randrange(days_between)
            accident_date = start_date + timedelta(days=random_days)

            # Generate time (accidents peak during 18-21 hours)
            hour_weights = [2, 1, 1, 1, 2, 4, 8, 12, 15, 12, 10, 12, 
                           15, 18, 20, 22, 25, 28, 30, 25, 20, 15, 10, 5]
            hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
            minute = random.randint(0, 59)

            accident_datetime = accident_date.replace(hour=hour, minute=minute)

            # Road type distribution
            road_types = ['National Highway', 'State Highway', 'Other Roads']
            road_type_probs = [0.329, 0.231, 0.440]  # Based on MORTH data
            road_type = np.random.choice(road_types, p=road_type_probs)

            # Accident severity
            severity_types = ['Fatal', 'Grievous Injury', 'Minor Injury', 'Non-Injury']
            severity_probs = [0.338, 0.311, 0.293, 0.058]
            severity = np.random.choice(severity_types, p=severity_probs)

            # Casualty numbers based on severity
            if severity == 'Fatal':
                deaths = random.randint(1, 4)
                injuries = random.randint(0, 3)
            elif severity == 'Grievous Injury':
                deaths = 0
                injuries = random.randint(1, 5)
            elif severity == 'Minor Injury':
                deaths = 0
                injuries = random.randint(1, 3)
            else:
                deaths = 0
                injuries = 0

            # Collision type
            collision_types = ['Hit from Back', 'Head on Collision', 'Hit and Run', 
                             'Hit from Side', 'Run off Road', 'Others']
            collision_probs = [0.214, 0.169, 0.146, 0.154, 0.045, 0.272]
            collision_type = np.random.choice(collision_types, p=collision_probs)

            # Traffic rule violation
            violations = ['Over Speeding', 'Driving on Wrong Side', 'Jumping Red Light',
                         'Drunken Driving', 'Mobile Phone Use', 'Others', 'None']
            violation_probs = [0.712, 0.054, 0.028, 0.018, 0.015, 0.123, 0.050]
            violation = np.random.choice(violations, p=violation_probs)

            # Vehicle type
            vehicle_types = ['Two Wheeler', 'Car/Taxi', 'Truck/Lorry', 'Bus', 
                           'Auto Rickshaw', 'Others']
            vehicle_probs = [0.445, 0.203, 0.168, 0.045, 0.067, 0.072]
            vehicle_type = np.random.choice(vehicle_types, p=vehicle_probs)

            # Weather condition
            weather_conditions = ['Clear/Sunny', 'Rainy', 'Foggy', 'Others']
            weather_probs = [0.78, 0.12, 0.06, 0.04]
            weather = np.random.choice(weather_conditions, p=weather_probs)

            # Road condition
            road_conditions = ['Dry', 'Wet', 'Others']
            road_cond_probs = [0.82, 0.15, 0.03]
            road_condition = np.random.choice(road_conditions, p=road_cond_probs)

            # Junction type
            junction_types = ['Straight Road', 'T-Junction', 'Y-Junction', 
                            'Four-arm Junction', 'Others']
            junction_probs = [0.67, 0.15, 0.08, 0.06, 0.04]
            junction_type = np.random.choice(junction_types, p=junction_probs)

            # Area type
            area_types = ['Rural', 'Urban']
            area_probs = [0.68, 0.32]  # Rural areas have more accidents
            area_type = np.random.choice(area_types, p=area_probs)

            # Environmental factors
            temperature = random.uniform(15, 45)  # Celsius
            humidity = random.uniform(30, 90)     # Percentage
            rainfall = random.exponential(0.5) if weather == 'Rainy' else 0
            wind_speed = random.uniform(5, 25)    # km/h
            visibility = random.uniform(0.5, 10) if weather == 'Foggy' else random.uniform(8, 15)

            record = {
                'accident_id': f"ACC_{i+1:06d}",
                'date': accident_datetime.date(),
                'time': accident_datetime.time(),
                'datetime': accident_datetime,
                'state': state,
                'district': f"District_{random.randint(1, 50)}",
                'area_type': area_type,
                'road_type': road_type,
                'junction_type': junction_type,
                'collision_type': collision_type,
                'severity': severity,
                'deaths': deaths,
                'injuries': injuries,
                'total_casualties': deaths + injuries,
                'vehicle_type': vehicle_type,
                'traffic_violation': violation,
                'weather_condition': weather,
                'road_condition': road_condition,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'rainfall': round(rainfall, 2),
                'wind_speed': round(wind_speed, 1),
                'visibility': round(visibility, 1),
                'hour': hour,
                'day_of_week': accident_datetime.weekday(),
                'month': accident_datetime.month,
                'season': self.get_season(accident_datetime.month),
                'is_weekend': 1 if accident_datetime.weekday() >= 5 else 0,
                'is_night': 1 if hour < 6 or hour > 20 else 0,
                'is_peak_hour': 1 if hour in [7, 8, 17, 18, 19, 20] else 0
            }

            data.append(record)

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic accident records")
        return df

    def get_season(self, month: int) -> str:
        """Get season based on month (Indian seasons)"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'

    def preprocess_accident_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess accident data for machine learning"""

        # Create additional features
        df['accident_severity_score'] = df['severity'].map({
            'Fatal': 4,
            'Grievous Injury': 3,
            'Minor Injury': 2,
            'Non-Injury': 1
        })

        # Create risk score based on multiple factors
        df['risk_score'] = (
            df['accident_severity_score'] * 0.4 +
            df['total_casualties'] * 0.3 +
            (df['is_night'] * 2) * 0.1 +
            (df['is_weekend'] * 1.5) * 0.1 +
            (df['weather_condition'] == 'Rainy').astype(int) * 0.1
        )

        # Encode categorical variables
        categorical_cols = ['state', 'road_type', 'collision_type', 'vehicle_type', 
                           'traffic_violation', 'weather_condition', 'junction_type']

        for col in categorical_cols:
            df[f'{col}_encoded'] = pd.Categorical(df[col]).codes

        # Create target variable (binary: accident occurred = 1)
        df['accident_occurred'] = 1

        logger.info("Accident data preprocessing completed")
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save processed accident data"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"processed_accident_data_{timestamp}.csv"

        filepath = os.path.join(self.processed_data_path, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Processed accident data saved to {filepath}")
        return filepath

    def get_accident_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate accident statistics summary"""
        stats = {
            'total_accidents': len(df),
            'total_deaths': df['deaths'].sum(),
            'total_injuries': df['injuries'].sum(),
            'fatality_rate': (df['deaths'].sum() / len(df)) * 100,
            'accident_by_state': df['state'].value_counts().to_dict(),
            'accident_by_severity': df['severity'].value_counts().to_dict(),
            'accident_by_road_type': df['road_type'].value_counts().to_dict(),
            'accident_by_hour': df['hour'].value_counts().sort_index().to_dict(),
            'accident_by_month': df['month'].value_counts().sort_index().to_dict(),
            'most_common_violations': df['traffic_violation'].value_counts().head().to_dict(),
            'weather_impact': df.groupby('weather_condition')['total_casualties'].mean().to_dict()
        }

        return stats

# Main execution
if __name__ == "__main__":
    processor = AccidentDataProcessor()

    print("Generating synthetic Indian road accident data...")
    accident_df = processor.generate_synthetic_accident_data(10000)

    print("Preprocessing accident data...")
    processed_df = processor.preprocess_accident_data(accident_df)

    # Save processed data
    filepath = processor.save_processed_data(processed_df)
    print(f"Processed data saved: {filepath}")

    # Generate statistics
    stats = processor.get_accident_statistics(processed_df)

    print("\n=== ACCIDENT STATISTICS SUMMARY ===")
    print(f"Total Accidents: {stats['total_accidents']:,}")
    print(f"Total Deaths: {stats['total_deaths']:,}")
    print(f"Total Injuries: {stats['total_injuries']:,}")
    print(f"Fatality Rate: {stats['fatality_rate']:.2f}%")

    print("\nTop 5 States by Accidents:")
    for state, count in list(stats['accident_by_state'].items())[:5]:
        print(f"  {state}: {count:,}")

    print("\nAccidents by Time of Day (Peak Hours):")
    sorted_hours = sorted(stats['accident_by_hour'].items(), key=lambda x: x[1], reverse=True)
    for hour, count in sorted_hours[:5]:
        print(f"  {hour:02d}:00 - {count:,} accidents")

    print(f"\nData preview:")
    print(processed_df[['date', 'state', 'severity', 'deaths', 'injuries', 'weather_condition']].head())
