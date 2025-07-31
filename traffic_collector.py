"""
Traffic Data Collector for Indian Traffic Accident Prediction
Collects real-time traffic data from various Indian APIs
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

class TrafficDataCollector:
    """Collects traffic data from various Indian APIs"""

    def __init__(self):
        self.google_api_key = API_KEYS.get('GOOGLE_MAPS_API_KEY')
        self.mapmyindia_key = API_KEYS.get('MAPMYINDIA_API_KEY')
        self.session = requests.Session()

    def get_google_traffic_data(self, origin: str, destination: str) -> Optional[Dict]:
        """
        Get traffic data from Google Maps Distance Matrix API
        """
        try:
            if not self.google_api_key or self.google_api_key == 'YOUR_GOOGLE_MAPS_API_KEY_HERE':
                logger.warning("Google Maps API key not configured")
                return self._generate_mock_traffic_data(origin, destination)

            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                'origins': origin,
                'destinations': destination,
                'departure_time': 'now',
                'traffic_model': 'best_guess',
                'key': self.google_api_key
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data['status'] == 'OK' and data['rows'][0]['elements'][0]['status'] == 'OK':
                element = data['rows'][0]['elements'][0]

                traffic_data = {
                    'origin': origin,
                    'destination': destination,
                    'distance_km': element['distance']['value'] / 1000,
                    'duration_normal_min': element['duration']['value'] / 60,
                    'duration_traffic_min': element.get('duration_in_traffic', {}).get('value', 0) / 60,
                    'traffic_delay_min': (element.get('duration_in_traffic', {}).get('value', 0) - element['duration']['value']) / 60,
                    'timestamp': datetime.now().isoformat()
                }

                return traffic_data
            else:
                logger.error(f"API returned error: {data.get('status', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Error fetching Google traffic data: {e}")
            return self._generate_mock_traffic_data(origin, destination)

    def _generate_mock_traffic_data(self, origin: str, destination: str) -> Dict:
        """Generate mock traffic data when API is not available"""
        import random

        base_distance = random.uniform(10, 50)  # km
        base_duration = base_distance * random.uniform(1.5, 3.0)  # minutes
        traffic_multiplier = random.uniform(1.1, 2.5)

        return {
            'origin': origin,
            'destination': destination,
            'distance_km': round(base_distance, 2),
            'duration_normal_min': round(base_duration, 2),
            'duration_traffic_min': round(base_duration * traffic_multiplier, 2),
            'traffic_delay_min': round(base_duration * (traffic_multiplier - 1), 2),
            'timestamp': datetime.now().isoformat(),
            'data_source': 'mock'
        }

    def collect_traffic_matrix(self, locations: List[str]) -> pd.DataFrame:
        """Collect traffic data between multiple locations"""
        traffic_data = []

        for i, origin in enumerate(locations):
            for j, destination in enumerate(locations):
                if i != j:  # Don't collect data for same origin-destination
                    data = self.get_google_traffic_data(origin, destination)
                    if data:
                        traffic_data.append(data)

        return pd.DataFrame(traffic_data)

    def get_accident_hotspots(self) -> pd.DataFrame:
        """
        Generate accident hotspot data based on known high-risk areas in India
        In a real implementation, this would fetch from accident databases
        """
        import random

        hotspots = [
            {"location": "Delhi-Gurgaon Expressway", "state": "Delhi", "risk_level": "High"},
            {"location": "Mumbai-Pune Highway", "state": "Maharashtra", "risk_level": "Very High"},
            {"location": "Bangalore-Mysore Road", "state": "Karnataka", "risk_level": "High"},
            {"location": "Chennai-Bangalore Highway", "state": "Tamil Nadu", "risk_level": "High"},
            {"location": "Hyderabad Outer Ring Road", "state": "Telangana", "risk_level": "Medium"},
            {"location": "Kolkata-Durgapur Highway", "state": "West Bengal", "risk_level": "High"},
            {"location": "Ahmedabad-Vadodara Highway", "state": "Gujarat", "risk_level": "Medium"},
            {"location": "Jaipur-Delhi Highway", "state": "Rajasthan", "risk_level": "High"}
        ]

        # Add some random data
        for hotspot in hotspots:
            hotspot.update({
                'accidents_last_month': random.randint(15, 85),
                'fatalities_last_month': random.randint(2, 25),
                'average_traffic_volume': random.randint(5000, 25000),
                'weather_conditions': random.choice(['Clear', 'Rainy', 'Foggy', 'Humid']),
                'timestamp': datetime.now().isoformat()
            })

        return pd.DataFrame(hotspots)

    def save_traffic_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save traffic data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"traffic_data_{timestamp}.csv"

        filepath = os.path.join('data/raw', filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Traffic data saved to {filepath}")
        return filepath

# Main execution
if __name__ == "__main__":
    collector = TrafficDataCollector()

    # Major Indian cities for traffic analysis
    major_cities = [
        "Delhi, India",
        "Mumbai, India", 
        "Bangalore, India",
        "Chennai, India",
        "Kolkata, India",
        "Hyderabad, India",
        "Pune, India",
        "Ahmedabad, India"
    ]

    print("Collecting traffic data between major Indian cities...")

    # Collect traffic matrix (this will take some time)
    traffic_df = collector.collect_traffic_matrix(major_cities[:4])  # Start with first 4 cities

    if not traffic_df.empty:
        filepath = collector.save_traffic_data(traffic_df)
        print(f"Traffic data collected and saved: {filepath}")
        print(traffic_df.head())

    # Collect accident hotspot data
    print("\nGenerating accident hotspot data...")
    hotspots_df = collector.get_accident_hotspots()
    hotspots_filepath = collector.save_traffic_data(hotspots_df, "accident_hotspots.csv")
    print(f"Hotspot data saved: {hotspots_filepath}")
    print(hotspots_df.head())
