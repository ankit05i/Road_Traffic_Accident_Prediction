# Indian Road Traffic Accident Prediction System

A comprehensive machine learning system for predicting road traffic accidents in India using real-time data, featuring Tableau integration for interactive dashboards and analytics.

## 🌟 Features

- **Real-time Data Collection**: Collects weather data from IMD (India Meteorological Department) and traffic data from various APIs
- **Advanced ML Models**: Multiple machine learning algorithms (Random Forest, XGBoost, LightGBM, Logistic Regression)
- **Tableau Integration**: Complete TabPy integration for interactive dashboards and real-time predictions
- **Indian Data Focus**: Uses official Indian government data sources and patterns from MORTH reports
- **Interactive Dashboards**: Pre-configured Tableau workbook with KPIs, filters, and predictive analytics
- **Comprehensive Analysis**: Weather impact, time-based patterns, geographical analysis, and risk assessment

## 📊 Data Sources

### Real-time APIs
- **India Meteorological Department (IMD)**: Weather data, rainfall, temperature, humidity
- **Google Maps API**: Traffic conditions and travel times
- **MapMyIndia API**: Indian traffic and road data
- **Government Open Data APIs**: Official accident statistics

### Processed Data
- Historical accident data based on MORTH (Ministry of Road Transport & Highways) reports
- State-wise accident patterns and statistics
- Weather correlation with accident frequency
- Time-based accident analysis (hourly, daily, seasonal)

## 🏗️ Project Structure

```
Road_Traffic_Accident_Prediction/
├── data/
│   ├── raw/                    # Raw data from APIs
│   └── processed/              # Cleaned and processed data
├── src/
│   ├── data_collection/        # Data collection scripts
│   │   ├── weather_collector.py
│   │   ├── traffic_collector.py
│   │   └── accident_processor.py
│   ├── preprocessing/          # Data preprocessing utilities
│   ├── models/                 # Machine learning models
│   │   └── ml_models.py
│   ├── tableau_integration/    # Tableau and TabPy integration
│   │   └── tabpy_functions.py
│   └── visualization/          # Additional visualization tools
├── notebooks/                  # Jupyter notebooks for analysis
├── config/                     # Configuration files
│   └── config.py
├── tableau_workbooks/          # Tableau workbook specifications
├── documentation/              # Project documentation
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Road_Traffic_Accident_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Copy the environment template and add your API keys:

```bash
cp .env.template .env
```

Edit `.env` file with your API keys:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
WEATHER_UNION_API_KEY=your_weather_union_api_key_here
MAPMYINDIA_API_KEY=your_mapmyindia_api_key_here
```

### 3. Run the System

Execute the complete pipeline:

```bash
python main.py
```

This will:
- Collect real-time weather and traffic data
- Generate synthetic accident data based on Indian patterns
- Train multiple machine learning models
- Create Tableau data extracts
- Setup TabPy integration
- Generate documentation and instructions

### 4. Tableau Setup

1. **Install Tableau Desktop** (if not already installed)

2. **Start TabPy Server**:
   ```bash
   tabpy
   ```

3. **Follow Tableau Instructions**:
   - Open `documentation/tableau_instructions.md`
   - Follow step-by-step instructions to create interactive dashboards

## 📈 Machine Learning Models

The system implements and compares multiple ML algorithms:

### Models Included
- **Random Forest**: Ensemble method with feature importance analysis
- **XGBoost**: Gradient boosting with high accuracy
- **LightGBM**: Fast gradient boosting with efficient memory usage
- **Logistic Regression**: Linear model for interpretability

### Features Used
- **Weather Conditions**: Temperature, humidity, rainfall, wind speed, visibility
- **Time Factors**: Hour, day of week, month, season, weekend/weekday
- **Road Characteristics**: Road type, junction type, traffic density
- **Location Data**: State, district, urban/rural classification
- **Environmental**: Weather conditions, road surface conditions

### Model Performance
All models are evaluated using:
- Accuracy scores
- Cross-validation metrics
- Classification reports
- Feature importance analysis

## 🎯 Tableau Dashboard Features

### Interactive Visualizations
1. **Accident Trends**: Time series analysis of accident patterns
2. **Geographical Analysis**: State-wise accident distribution map
3. **Weather Impact**: Weather conditions vs accident severity
4. **Time Pattern Analysis**: Heatmap of accidents by hour and day
5. **Road Type Distribution**: Accident breakdown by road categories
6. **Real-time Prediction**: Interactive prediction based on current conditions

### Key Performance Indicators (KPIs)
- Total accidents and trends
- Fatality rates by region
- Most dangerous times and locations
- Weather impact factors
- Accident prediction confidence scores

### Filters and Parameters
- Date range selection
- State and region filters
- Severity level filters
- Weather condition filters
- Interactive prediction parameters

## 🔧 Configuration

### API Configuration
Edit `config/config.py` to modify:
- API endpoints and keys
- Data collection parameters
- Model hyperparameters
- Tableau connection settings

### Model Configuration
```python
MODEL_CONFIG = {
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'N_ESTIMATORS': 100,
    'MAX_DEPTH': 10,
    'LEARNING_RATE': 0.1
}
```

### Feature Configuration
Customize which features to include in model training:
```python
FEATURE_CONFIG = {
    'WEATHER_FEATURES': ['temperature', 'humidity', 'rainfall', 'wind_speed'],
    'TIME_FEATURES': ['hour', 'day_of_week', 'month', 'season'],
    'ROAD_FEATURES': ['road_type', 'speed_limit', 'traffic_density'],
    'LOCATION_FEATURES': ['state', 'district', 'urban_rural']
}
```

## 📊 Data Analysis

### Included Jupyter Notebooks
- `01_Data_Exploration.ipynb`: Exploratory data analysis
- `02_Weather_Analysis.ipynb`: Weather pattern analysis
- `03_Time_Series_Analysis.ipynb`: Temporal accident patterns
- `04_Model_Comparison.ipynb`: ML model performance comparison
- `05_Tableau_Preparation.ipynb`: Data preparation for Tableau

### Statistics Generated
- Accident frequency by state
- Seasonal and temporal patterns
- Weather correlation analysis
- Road type risk assessment
- Demographic impact analysis

## 🔍 Usage Examples

### Collecting Real-time Data
```python
from src.data_collection.weather_collector import WeatherDataCollector

collector = WeatherDataCollector()
weather_data = collector.get_current_weather("42182")  # Delhi
print(weather_data)
```

### Making Predictions
```python
from src.models.ml_models import AccidentPredictionModel

model = AccidentPredictionModel()
model.load_models('models/accident_prediction_models.joblib')

prediction = model.predict_accident_risk({
    'temperature': 35.0,
    'humidity': 80.0,
    'rainfall': 5.0,
    'hour': 20,
    'state': 'Tamil Nadu',
    'road_type': 'National Highway'
})
```

### Tableau Integration
```python
from src.tableau_integration.tabpy_functions import TableauIntegration

integration = TableauIntegration()
integration.setup_tabpy_functions()
```

## 📝 Important Notes

### Data Sources
- Uses official Indian government data patterns
- Real-time APIs require proper authentication
- Synthetic data generation follows MORTH statistical patterns
- Weather data sourced from India Meteorological Department

### Tableau Requirements
- Tableau Desktop 2019.1 or later
- TabPy server running on localhost:9004
- Proper network access to APIs
- Sufficient system resources for data processing

### Performance Considerations
- Large dataset processing may require substantial RAM
- API rate limits may affect real-time data collection
- Model training time varies based on dataset size
- Tableau dashboard performance depends on data volume

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation folder for detailed guides
- Review Tableau integration instructions

## 🙏 Acknowledgments

- Ministry of Road Transport and Highways (MORTH) for official accident statistics
- India Meteorological Department (IMD) for weather data APIs
- Tableau community for TabPy integration examples
- Open source ML libraries: scikit-learn, XGBoost, LightGBM

## 🔄 Updates

- **Version 1.0**: Initial release with basic prediction models
- **Version 1.1**: Added Tableau integration and real-time APIs
- **Version 1.2**: Enhanced with interactive dashboards and KPIs

---

**Built with ❤️ for road safety in India**

For detailed setup instructions, see `documentation/tableau_instructions.md`
