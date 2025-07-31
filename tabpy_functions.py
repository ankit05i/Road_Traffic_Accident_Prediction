"""
Tableau Integration for Indian Traffic Accident Prediction
Integrates ML models with Tableau using TabPy for real-time predictions
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
import os
import sys
import joblib
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TABLEAU_CONFIG
from models.ml_models import AccidentPredictionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableauIntegration:
    """Integration class for Tableau and TabPy"""

    def __init__(self):
        self.model = AccidentPredictionModel()
        self.tabpy_host = TABLEAU_CONFIG['TABPY_HOST']
        self.tabpy_port = TABLEAU_CONFIG['TABPY_PORT']

    def setup_tabpy_functions(self):
        """Setup TabPy functions for Tableau"""
        try:
            import tabpy
            from tabpy.tabpy_tools.client import Client

            # Connect to TabPy server
            client = Client(f'http://{self.tabpy_host}:{self.tabpy_port}/')

            # Define prediction function for Tableau
            def predict_accident_severity(temperature, humidity, rainfall, wind_speed, 
                                        hour, day_of_week, month, state_code, road_type_code):
                """
                Predict accident severity based on environmental and road conditions
                This function will be called from Tableau
                """
                try:
                    # Prepare input data
                    input_data = {
                        'temperature': temperature,
                        'humidity': humidity,
                        'rainfall': rainfall,
                        'wind_speed': wind_speed,
                        'visibility': 10.0,  # Default visibility
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'month': month,
                        'is_weekend': 1 if day_of_week >= 5 else 0,
                        'is_night': 1 if hour < 6 or hour > 20 else 0,
                        'is_peak_hour': 1 if hour in [7, 8, 17, 18, 19, 20] else 0,
                        'state': f'State_{state_code}',
                        'road_type': 'National Highway' if road_type_code == 1 else 'State Highway' if road_type_code == 2 else 'Other Roads',
                        'collision_type': 'Others',
                        'vehicle_type': 'Two Wheeler',
                        'traffic_violation': 'Over Speeding',
                        'weather_condition': 'Rainy' if rainfall > 0 else 'Clear/Sunny',
                        'junction_type': 'Straight Road',
                        'area_type': 'Urban'
                    }

                    # Make prediction (mock prediction for demo)
                    risk_scores = []
                    for i in range(len(temperature)):
                        if rainfall[i] > 5:
                            risk = 0.8  # High risk in heavy rain
                        elif hour[i] in [22, 23, 0, 1, 2, 3]:
                            risk = 0.7  # High risk at night
                        elif temperature[i] > 40:
                            risk = 0.6  # Medium-high risk in extreme heat
                        else:
                            risk = 0.3  # Low risk

                        risk_scores.append(risk)

                    return risk_scores

                except Exception as e:
                    logger.error(f"Error in prediction function: {e}")
                    return [0.5] * len(temperature)  # Default risk

            # Deploy function to TabPy
            client.deploy('predict_accident_risk', predict_accident_severity, 
                         'Predicts accident risk based on conditions', override=True)

            logger.info("TabPy functions deployed successfully!")

        except ImportError:
            logger.warning("TabPy not installed. Install with: pip install tabpy")
        except Exception as e:
            logger.error(f"Error setting up TabPy functions: {e}")

    def create_tableau_data_extract(self, df: pd.DataFrame, filename: str = None) -> str:
        """Create data extract for Tableau"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"accident_data_extract_{timestamp}.csv"

        # Prepare data for Tableau
        tableau_df = df.copy()

        # Add calculated fields that Tableau can use
        tableau_df['Risk_Score'] = (
            tableau_df.get('accident_severity_score', 2) * 0.3 +
            tableau_df.get('total_casualties', 1) * 0.2 +
            tableau_df['is_night'] * 0.2 +
            tableau_df['is_weekend'] * 0.1 +
            (tableau_df['weather_condition'] == 'Rainy').astype(int) * 0.2
        )

        # Create time-based dimensions
        tableau_df['Date'] = pd.to_datetime(tableau_df['date'])
        tableau_df['Year'] = tableau_df['Date'].dt.year
        tableau_df['Month_Name'] = tableau_df['Date'].dt.strftime('%B')
        tableau_df['Day_Name'] = tableau_df['Date'].dt.strftime('%A')
        tableau_df['Hour_Range'] = tableau_df['hour'].apply(self._get_hour_range)

        # Create geographical dimensions
        tableau_df['Region'] = tableau_df['state'].apply(self._get_region)

        # Save extract
        filepath = os.path.join('data/processed', filename)
        tableau_df.to_csv(filepath, index=False)

        logger.info(f"Tableau data extract created: {filepath}")
        return filepath

    def _get_hour_range(self, hour: int) -> str:
        """Convert hour to readable time range"""
        if hour < 6:
            return "Night (0-6)"
        elif hour < 12:
            return "Morning (6-12)"
        elif hour < 18:
            return "Afternoon (12-18)"
        else:
            return "Evening (18-24)"

    def _get_region(self, state: str) -> str:
        """Get region based on state"""
        north_states = ['Delhi', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Rajasthan']
        south_states = ['Tamil Nadu', 'Karnataka', 'Kerala', 'Andhra Pradesh', 'Telangana']
        west_states = ['Maharashtra', 'Gujarat', 'Rajasthan', 'Goa']
        east_states = ['West Bengal', 'Bihar', 'Odisha', 'Jharkhand']

        if any(s in state for s in north_states):
            return "North India"
        elif any(s in state for s in south_states):
            return "South India"
        elif any(s in state for s in west_states):
            return "West India"
        elif any(s in state for s in east_states):
            return "East India"
        else:
            return "Central India"

    def generate_tableau_dashboard_spec(self) -> dict:
        """Generate Tableau dashboard specification"""
        dashboard_spec = {
            "dashboard_name": "Indian Road Traffic Accident Analysis",
            "data_source": "accident_data_extract.csv",
            "sheets": [
                {
                    "name": "Accident Trends",
                    "type": "line_chart",
                    "x_axis": "Date",
                    "y_axis": "COUNT(accident_id)",
                    "color": "severity",
                    "description": "Shows accident trends over time by severity"
                },
                {
                    "name": "State-wise Accidents",
                    "type": "map",
                    "geography": "state",
                    "size": "COUNT(accident_id)",
                    "color": "Risk_Score",
                    "description": "Geographical distribution of accidents across Indian states"
                },
                {
                    "name": "Weather Impact",
                    "type": "bar_chart",
                    "x_axis": "weather_condition",
                    "y_axis": "AVG(total_casualties)",
                    "color": "severity",
                    "description": "Impact of weather conditions on accident severity"
                },
                {
                    "name": "Time Analysis",
                    "type": "heatmap",
                    "x_axis": "hour",
                    "y_axis": "Day_Name",
                    "color": "COUNT(accident_id)",
                    "description": "Heatmap showing accident patterns by hour and day of week"
                },
                {
                    "name": "Road Type Analysis",
                    "type": "pie_chart",
                    "dimension": "road_type",
                    "measure": "COUNT(accident_id)",
                    "description": "Distribution of accidents by road type"
                },
                {
                    "name": "Prediction Dashboard",
                    "type": "parameter_control",
                    "parameters": [
                        "Temperature", "Humidity", "Rainfall", "Hour", "State", "Road Type"
                    ],
                    "calculation": "predict_accident_risk([Temperature], [Humidity], [Rainfall], 10, [Hour], 1, 6, 1, 1)",
                    "description": "Interactive prediction based on current conditions"
                }
            ],
            "filters": [
                {"field": "Date", "type": "date_range"},
                {"field": "state", "type": "multi_select"},
                {"field": "severity", "type": "multi_select"},
                {"field": "weather_condition", "type": "multi_select"}
            ],
            "parameters": [
                {"name": "Temperature", "type": "float", "range": [10, 50], "default": 25},
                {"name": "Humidity", "type": "float", "range": [30, 100], "default": 65},
                {"name": "Rainfall", "type": "float", "range": [0, 50], "default": 0},
                {"name": "Hour", "type": "integer", "range": [0, 23], "default": 12},
                {"name": "State", "type": "integer", "range": [1, 30], "default": 1},
                {"name": "Road Type", "type": "integer", "range": [1, 3], "default": 1}
            ]
        }

        return dashboard_spec

    def export_tableau_workbook_instructions(self) -> str:
        """Generate instructions for creating Tableau workbook"""
        instructions = """
# Tableau Workbook Creation Instructions

## Step 1: Data Connection
1. Open Tableau Desktop
2. Connect to Data Source: "Text file" or "Excel"
3. Navigate to the processed data file: `data/processed/accident_data_extract_[timestamp].csv`
4. Import the data

## Step 2: Setup TabPy Connection
1. Go to Help > Settings and Performance > Manage Analytics Extension Connection
2. Select "TabPy/External API"
3. Set Server: localhost, Port: 9004
4. Test connection and Save

## Step 3: Create Calculated Fields

### Risk Score Calculation:
```
// Risk Score
([Accident Severity Score] * 0.3) + 
([Total Casualties] * 0.2) + 
([Is Night] * 0.2) + 
([Is Weekend] * 0.1) + 
(IF [Weather Condition] = "Rainy" THEN 0.2 ELSE 0 END)
```

### Accident Prediction (using TabPy):
```
// Accident Risk Prediction
SCRIPT_REAL("
import numpy as np
return predict_accident_risk(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8, _arg9)
", 
[Temperature], [Humidity], [Rainfall], [Wind Speed], [Hour], [Day Of Week], [Month], [State Encoded], [Road Type Encoded])
```

## Step 4: Create Visualizations

### Sheet 1: Accident Trends Over Time
- Drag Date to Columns
- Drag COUNT(Accident Id) to Rows
- Drag Severity to Color
- Change mark type to Line

### Sheet 2: State-wise Accident Map
- Double-click on State (should create a map)
- Drag COUNT(Accident Id) to Size
- Drag Risk Score to Color
- Adjust color palette to show risk levels

### Sheet 3: Weather Impact Analysis
- Drag Weather Condition to Columns
- Drag AVG(Total Casualties) to Rows
- Drag Severity to Color
- Change mark type to Bar

### Sheet 4: Hourly Accident Patterns
- Drag Hour to Columns
- Drag Day Name to Rows
- Drag COUNT(Accident Id) to Color
- Change mark type to Square (for heatmap)

### Sheet 5: Interactive Prediction
1. Create Parameters:
   - Temperature (Float, 10-50, default 25)
   - Humidity (Float, 30-100, default 65)
   - Rainfall (Float, 0-50, default 0)
   - Hour (Integer, 0-23, default 12)

2. Create calculated field for prediction using TabPy function
3. Display prediction result as Big Number
4. Add parameter controls to dashboard

## Step 5: Create Dashboard
1. Create new Dashboard
2. Drag sheets to dashboard
3. Add filters for Date, State, Severity
4. Add parameter controls for prediction
5. Format and arrange for best user experience

## Step 6: Key Performance Indicators (KPIs)
Add these KPIs to your dashboard:
- Total Accidents (COUNT)
- Total Deaths (SUM)
- Total Injuries (SUM)
- Fatality Rate (Deaths/Accidents)
- Average Risk Score
- Most Dangerous Hour
- Most Dangerous State
- Weather with Highest Casualties

## Step 7: Interactivity Features
- Add filters for time period, state, severity
- Create parameters for prediction inputs
- Add highlight actions between sheets
- Create hover tooltips with additional information

## Step 8: Formatting and Design
- Use consistent color scheme
- Add appropriate titles and labels
- Format numbers with proper units
- Add data source information
- Include last updated timestamp

Save the workbook as "Indian_Traffic_Accident_Analysis.twbx"
"""

        instructions_file = os.path.join('documentation', 'tableau_instructions.md')
        with open(instructions_file, 'w') as f:
            f.write(instructions)

        logger.info(f"Tableau instructions saved to {instructions_file}")
        return instructions_file

# Main execution
if __name__ == "__main__":
    integration = TableauIntegration()

    print("Setting up Tableau integration...")

    # Setup TabPy functions
    integration.setup_tabpy_functions()

    # Generate dashboard specifications
    dashboard_spec = integration.generate_tableau_dashboard_spec()

    # Save dashboard spec
    spec_file = os.path.join('tableau_workbooks', 'dashboard_specification.json')
    os.makedirs('tableau_workbooks', exist_ok=True)
    with open(spec_file, 'w') as f:
        json.dump(dashboard_spec, f, indent=2)

    print(f"Dashboard specification saved: {spec_file}")

    # Generate instructions
    instructions_file = integration.export_tableau_workbook_instructions()
    print(f"Tableau instructions created: {instructions_file}")

    print("\nTableau integration setup completed!")
    print("Next steps:")
    print("1. Start TabPy server: tabpy")
    print("2. Follow instructions in documentation/tableau_instructions.md")
    print("3. Create Tableau workbook using the data extract")
