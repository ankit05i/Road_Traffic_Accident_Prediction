"""
Weather Data Collector for Indian Traffic Accident Prediction
Collects real-time weather data from IMD (India Meteorological Department) APIs
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import API_KEYS, DATA_SOURCES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Collects weather data from Indian Meteorological Department APIs"""

    def __init__(self):
        self.base_url = API_KEYS['IMD_API_BASE']
        self.session = requests.Session()

    def get_current_weather(self, station_id: str = "42182") -> Optional[Dict]:
        """
        Get current weather data from IMD API
        Default station_id is for Delhi
        """
        try:
            url = f"{self.base_url}current_wx_api.php"
            params = {'id': station_id}

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Process and standardize the data
            weather_data = {
                'station_id': station_id,
                'station_name': data.get('Station', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'temperature': float(data.get('Temperature', 0)) if data.get('Temperature') else None,
                'humidity': float(data.get('Humidity', 0)) if data.get('Humidity') else None,
                'wind_speed': float(data.get('Wind Speed', 0)) if data.get('Wind Speed') else None,
                'wind_direction': data.get('Wind Direction', 'Unknown'),
                'pressure': float(data.get('M.S.L.P', 0)) if data.get('M.S.L.P') else None,
                'rainfall_24h': float(data.get('Last 24 hrs Rainfall', 0)) if data.get('Last 24 hrs Rainfall') else None,
                'weather_code': data.get('Weather Code', 'Unknown'),
                'visibility': data.get('Nebulosity', 'Unknown')
            }

            logger.info(f"Successfully collected weather data for station {station_id}")
            return weather_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def get_district_rainfall(self, district_id: str = "164") -> Optional[Dict]:
        """Get district-wise rainfall data"""
        try:
            url = f"{self.base_url}districtwise_rainfall_api.php"
            params = {'id': district_id}

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            rainfall_data = {
                'district_id': district_id,
                'district_name': data.get('District', 'Unknown'),
                'date': data.get('Date', datetime.now().strftime('%Y-%m-%d')),
                'daily_actual': float(data.get('Daily Actual', 0)) if data.get('Daily Actual') else 0,
                'daily_normal': float(data.get('Daily Normal', 0)) if data.get('Daily Normal') else 0,
                'weekly_actual': float(data.get('Weekly Actual', 0)) if data.get('Weekly Actual') else 0,
                'cumulative_actual': float(data.get('Cumulative Actual', 0)) if data.get('Cumulative Actual') else 0,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Successfully collected rainfall data for district {district_id}")
            return rainfall_data

        except Exception as e:
            logger.error(f"Error fetching rainfall data: {e}")
            return None

    def collect_multiple_stations(self, station_ids: List[str]) -> pd.DataFrame:
        """Collect weather data from multiple stations"""
        weather_data = []

        for station_id in station_ids:
            data = self.get_current_weather(station_id)
            if data:
                weather_data.append(data)

        return pd.DataFrame(weather_data)

    def save_weather_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save weather data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"weather_data_{timestamp}.csv"

        filepath = os.path.join('data/raw', filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Weather data saved to {filepath}")
        return filepath

# Main execution
if __name__ == "__main__":
    collector = WeatherDataCollector()

    # Major Indian cities station IDs (example IDs - replace with actual ones)
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

    print("Collecting weather data from major Indian cities...")
    weather_df = collector.collect_multiple_stations(list(major_cities.values()))

    if not weather_df.empty:
        filepath = collector.save_weather_data(weather_df)
        print(f"Weather data collected and saved: {filepath}")
        print(weather_df.head())
    else:
        print("No weather data collected")
