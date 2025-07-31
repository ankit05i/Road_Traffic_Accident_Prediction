# API Configuration
API_KEYS = {
    'IMD_API_BASE': 'https://mausam.imd.gov.in/api/',
    'GOOGLE_MAPS_API_KEY': 'YOUR_GOOGLE_MAPS_API_KEY_HERE',
    'WEATHER_UNION_API_KEY': 'YOUR_WEATHER_UNION_API_KEY_HERE',
    'MAPMYINDIA_API_KEY': 'YOUR_MAPMYINDIA_API_KEY_HERE'
}

# Data Sources Configuration
DATA_SOURCES = {
    'MORTH_ACCIDENT_DATA_URL': 'https://morth.nic.in/sites/default/files/RA_2022_30_Oct.pdf',
    'IMD_WEATHER_API': 'https://mausam.imd.gov.in/api/current_wx_api.php',
    'DISTRICT_RAINFALL_API': 'https://mausam.imd.gov.in/api/districtwise_rainfall_api.php',
    'TRAFFIC_CHALLAN_API': 'https://echallan.parivahan.gov.in/index/accused-challan'
}

# Model Configuration
MODEL_CONFIG = {
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'VALIDATION_SIZE': 0.2,
    'N_ESTIMATORS': 100,
    'MAX_DEPTH': 10,
    'LEARNING_RATE': 0.1
}

# Tableau Configuration
TABLEAU_CONFIG = {
    'TABPY_HOST': 'localhost',
    'TABPY_PORT': 9004,
    'SERVER_URL': 'http://localhost:8000',
    'USERNAME': 'admin',
    'PASSWORD': 'admin'
}

# File Paths
PATHS = {
    'RAW_DATA': 'data/raw/',
    'PROCESSED_DATA': 'data/processed/',
    'MODELS': 'src/models/',
    'LOGS': 'logs/',
    'TABLEAU_WORKBOOKS': 'tableau_workbooks/'
}

# Feature Engineering
FEATURE_CONFIG = {
    'WEATHER_FEATURES': ['temperature', 'humidity', 'rainfall', 'wind_speed', 'visibility'],
    'TIME_FEATURES': ['hour', 'day_of_week', 'month', 'season'],
    'ROAD_FEATURES': ['road_type', 'speed_limit', 'traffic_density', 'junction_type'],
    'LOCATION_FEATURES': ['state', 'district', 'urban_rural', 'population_density']
}
