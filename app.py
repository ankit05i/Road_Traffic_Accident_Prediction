import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Road Accident Prediction India",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1e3d59;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 2px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Create synthetic Indian road accident data for demonstration
@st.cache_data
def load_accident_data():
    """Load synthetic Indian road accident data"""
    np.random.seed(42)
    n_samples = 3000

    states = ['Maharashtra', 'Tamil Nadu', 'Karnataka', 'Gujarat', 'Rajasthan', 
              'Uttar Pradesh', 'Madhya Pradesh', 'West Bengal', 'Delhi', 'Punjab']

    data = {
        'state': np.random.choice(states, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(1, 8, n_samples),  # 1=Monday, 7=Sunday
        'month': np.random.randint(1, 13, n_samples),
        'weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Cloudy'], n_samples, 
                                   p=[0.5, 0.25, 0.15, 0.1]),
        'road_type': np.random.choice(['National Highway', 'State Highway', 'City Road', 'Village Road'], 
                                    n_samples, p=[0.3, 0.25, 0.35, 0.1]),
        'vehicle_type': np.random.choice(['Car', 'Two Wheeler', 'Truck', 'Bus', 'Auto'], 
                                       n_samples, p=[0.35, 0.3, 0.15, 0.1, 0.1]),
        'speed_limit': np.random.choice([30, 40, 50, 60, 80, 100], n_samples),
        'temperature': np.random.normal(28, 8, n_samples),  # Celsius
        'humidity': np.random.randint(40, 90, n_samples),  # Percentage
        'traffic_density': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.4, 0.3])
    }

    # Create target variable (accident severity)
    # Higher probability of severe accidents with certain conditions
    severity_prob = np.random.random(n_samples)

    # Adjust probabilities based on conditions
    severity_prob += (data['weather'] == 'Rain').astype(int) * 0.2
    severity_prob += (data['weather'] == 'Fog').astype(int) * 0.3
    severity_prob += (np.array(data['hour']) > 22).astype(int) * 0.2  # Night time
    severity_prob += (np.array(data['hour']) < 6).astype(int) * 0.2   # Early morning
    severity_prob += (data['road_type'] == 'National Highway').astype(int) * 0.15
    severity_prob += (data['traffic_density'] == 'High').astype(int) * 0.1

    data['severity'] = ['Fatal' if p > 0.7 else 'Serious' if p > 0.4 else 'Minor' 
                       for p in severity_prob]

    return pd.DataFrame(data)

@st.cache_data
def get_weather_data_demo():
    """Simulate weather data from IMD API"""
    return {
        'temperature': np.random.normal(32, 5),
        'humidity': np.random.randint(60, 85),
        'weather_condition': np.random.choice(['Clear', 'Rain', 'Cloudy', 'Fog'], p=[0.4, 0.3, 0.2, 0.1]),
        'wind_speed': np.random.normal(10, 3)
    }

def train_model(data):
    """Train Random Forest model for accident prediction"""
    # Prepare features
    le_state = LabelEncoder()
    le_weather = LabelEncoder()
    le_road = LabelEncoder()
    le_vehicle = LabelEncoder()
    le_traffic = LabelEncoder()
    le_severity = LabelEncoder()

    X = data.copy()
    X['state_encoded'] = le_state.fit_transform(X['state'])
    X['weather_encoded'] = le_weather.fit_transform(X['weather'])
    X['road_type_encoded'] = le_road.fit_transform(X['road_type'])
    X['vehicle_type_encoded'] = le_vehicle.fit_transform(X['vehicle_type'])
    X['traffic_density_encoded'] = le_traffic.fit_transform(X['traffic_density'])

    y = le_severity.fit_transform(data['severity'])

    features = ['hour', 'day_of_week', 'month', 'speed_limit', 'temperature', 
                'humidity', 'state_encoded', 'weather_encoded', 'road_type_encoded', 
                'vehicle_type_encoded', 'traffic_density_encoded']

    X_features = X[features]

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, le_severity, (le_state, le_weather, le_road, le_vehicle, le_traffic)

def main():
    st.markdown('<h1 class="main-header">🚗 Road Traffic Accident Prediction System for India</h1>', 
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Dashboard", "Accident Prediction", "Data Analysis", "Weather Data", "About"])

    # Load data and train model
    with st.spinner("Loading data and training model..."):
        data = load_accident_data()
        model, accuracy, le_severity, encoders = train_model(data)

    if page == "Dashboard":
        st.header("📊 Accident Statistics Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Accidents", len(data))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            fatal_count = len(data[data['severity'] == 'Fatal'])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Fatal Accidents", fatal_count)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            high_risk_hours = data[data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])]
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Night Accidents", len(high_risk_hours))
            st.markdown('</div>', unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Accidents by Severity")
            severity_counts = data['severity'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        color_discrete_map={'Fatal': '#ff4444', 'Serious': '#ffaa00', 'Minor': '#44ff44'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Accidents by Hour")
            hourly_accidents = data.groupby('hour').size()
            fig = px.bar(x=hourly_accidents.index, y=hourly_accidents.values,
                        labels={'x': 'Hour of Day', 'y': 'Number of Accidents'})
            fig.update_traces(marker_color='lightblue')
            st.plotly_chart(fig, use_container_width=True)

        # State-wise analysis
        st.subheader("State-wise Accident Analysis")
        state_analysis = data.groupby('state').agg({
            'severity': 'count',
        }).rename(columns={'severity': 'total_accidents'})

        state_analysis['fatal_accidents'] = data[data['severity'] == 'Fatal'].groupby('state').size().fillna(0)
        state_analysis['fatal_rate'] = (state_analysis['fatal_accidents'] / state_analysis['total_accidents'] * 100).round(2)

        fig = px.bar(state_analysis, x=state_analysis.index, y='total_accidents',
                    hover_data=['fatal_rate'], 
                    labels={'x': 'State', 'y': 'Total Accidents', 'fatal_rate': 'Fatal Rate (%)'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Accident Prediction":
        st.header("🔮 Real-time Accident Risk Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Location & Time")
            state = st.selectbox("State", 
                               ['Maharashtra', 'Tamil Nadu', 'Karnataka', 'Gujarat', 'Rajasthan', 
                                'Uttar Pradesh', 'Madhya Pradesh', 'West Bengal', 'Delhi', 'Punjab'])

            current_time = datetime.now()
            hour = st.slider("Hour of Day", 0, 23, current_time.hour)
            day_of_week = st.slider("Day of Week (1=Monday)", 1, 7, current_time.weekday() + 1)
            month = st.slider("Month", 1, 12, current_time.month)

        with col2:
            st.subheader("Road & Weather Conditions")
            weather = st.selectbox("Weather Condition", ['Clear', 'Rain', 'Fog', 'Cloudy'])
            road_type = st.selectbox("Road Type", ['National Highway', 'State Highway', 'City Road', 'Village Road'])
            vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Two Wheeler', 'Truck', 'Bus', 'Auto'])
            speed_limit = st.selectbox("Speed Limit (km/h)", [30, 40, 50, 60, 80, 100])

            temperature = st.slider("Temperature (°C)", 10, 45, 28)
            humidity = st.slider("Humidity (%)", 30, 95, 70)
            traffic_density = st.selectbox("Traffic Density", ['Low', 'Medium', 'High'])

        if st.button("Predict Accident Risk", type="primary"):
            # Encode inputs
            le_state, le_weather, le_road, le_vehicle, le_traffic = encoders

            try:
                state_encoded = le_state.transform([state])[0]
                weather_encoded = le_weather.transform([weather])[0]
                road_encoded = le_road.transform([road_type])[0]
                vehicle_encoded = le_vehicle.transform([vehicle_type])[0]
                traffic_encoded = le_traffic.transform([traffic_density])[0]

                # Make prediction
                features = np.array([[hour, day_of_week, month, speed_limit, temperature, 
                                    humidity, state_encoded, weather_encoded, road_encoded, 
                                    vehicle_encoded, traffic_encoded]])

                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]

                severity_label = le_severity.inverse_transform([prediction])[0]

                # Display result
                if severity_label == 'Fatal':
                    st.markdown(f'<div class="prediction-result high-risk">⚠️ HIGH RISK: {severity_label} Accident Predicted</div>', 
                               unsafe_allow_html=True)
                elif severity_label == 'Serious':
                    st.markdown(f'<div class="prediction-result medium-risk">⚡ MEDIUM RISK: {severity_label} Accident Predicted</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-result low-risk">✅ LOW RISK: {severity_label} Accident Predicted</div>', 
                               unsafe_allow_html=True)

                # Show probabilities
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Severity': le_severity.classes_,
                    'Probability': probability
                })

                fig = px.bar(prob_df, x='Severity', y='Probability', 
                           color='Severity',
                           color_discrete_map={'Fatal': '#ff4444', 'Serious': '#ffaa00', 'Minor': '#44ff44'})
                st.plotly_chart(fig, use_container_width=True)

                # Risk factors
                st.subheader("Risk Factors Analysis")
                risk_factors = []

                if weather in ['Rain', 'Fog']:
                    risk_factors.append(f"🌧️ Weather condition: {weather}")
                if hour >= 22 or hour <= 5:
                    risk_factors.append(f"🌙 Night time driving (Hour: {hour})")
                if road_type == 'National Highway':
                    risk_factors.append(f"🛣️ High-speed road: {road_type}")
                if traffic_density == 'High':
                    risk_factors.append("🚦 High traffic density")
                if vehicle_type == 'Two Wheeler':
                    risk_factors.append("🏍️ Vulnerable vehicle type: Two Wheeler")

                if risk_factors:
                    st.warning("Identified Risk Factors:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No major risk factors identified for current conditions.")

            except ValueError as e:
                st.error(f"Error in prediction: {str(e)}. Some values might not be in training data.")

    elif page == "Data Analysis":
        st.header("📈 Data Analysis & Insights")

        # Weather vs Accidents
        st.subheader("Weather Impact on Accidents")
        weather_analysis = data.groupby(['weather', 'severity']).size().unstack(fill_value=0)
        fig = px.bar(weather_analysis, barmode='group', 
                    title="Accident Severity by Weather Condition")
        st.plotly_chart(fig, use_container_width=True)

        # Time analysis
        st.subheader("Time-based Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Day of week analysis
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_accidents = data.groupby('day_of_week').size()
            day_accidents.index = [days[i-1] for i in day_accidents.index]

            fig = px.bar(x=day_accidents.index, y=day_accidents.values,
                        title="Accidents by Day of Week")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Monthly analysis
            month_accidents = data.groupby('month').size()
            fig = px.line(x=month_accidents.index, y=month_accidents.values,
                         title="Accidents by Month")
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("Feature Correlation Analysis")
        numeric_data = data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Weather Data":
        st.header("🌤️ Real-time Weather Data")

        st.info("This section demonstrates integration with IMD Weather APIs")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Fetch Current Weather Data", type="primary"):
                with st.spinner("Fetching weather data..."):
                    weather_data = get_weather_data_demo()

                st.success("Weather data fetched successfully!")

                st.metric("Temperature", f"{weather_data['temperature']:.1f}°C")
                st.metric("Humidity", f"{weather_data['humidity']}%")
                st.metric("Weather Condition", weather_data['weather_condition'])
                st.metric("Wind Speed", f"{weather_data['wind_speed']:.1f} km/h")

        with col2:
            st.subheader("API Information")
            st.write("""
            **Real APIs that can be integrated:**

            1. **IMD Weather API**
               - Current weather conditions
               - 7-day forecasts
               - District-wise rainfall data

            2. **Data Sources**
               - India Meteorological Department
               - Ministry of Road Transport & Highways
               - Open Government Data Platform

            3. **Features**
               - Real-time weather updates
               - Historical data analysis
               - Automated data collection
            """)

    else:  # About page
        st.header("ℹ️ About This Application")

        st.write("""
        ## Road Traffic Accident Prediction System for India

        This application uses machine learning to predict road accident severity based on various factors
        including weather conditions, road types, time of day, and geographic location.

        ### Features:
        - **Real-time Prediction**: Get instant accident risk assessment
        - **Indian Data**: Specifically designed for Indian road conditions
        - **Weather Integration**: Uses IMD weather data for predictions
        - **Interactive Dashboard**: Comprehensive data visualization
        - **State-wise Analysis**: Regional accident pattern analysis

        ### Technology Stack:
        - **Frontend**: Streamlit
        - **Machine Learning**: Random Forest, Scikit-learn
        - **Visualization**: Plotly
        - **Data Sources**: Synthetic data based on MoRTH reports

        ### Model Performance:
        - **Algorithm**: Random Forest Classifier
        - **Accuracy**: {:.2%}
        - **Features**: 11 input parameters
        - **Training Data**: 3,000 accident records

        ### Data Sources:
        1. Ministry of Road Transport & Highways (MoRTH)
        2. India Meteorological Department (IMD)
        3. Open Government Data Platform India
        4. State Transport Departments

        ### Disclaimer:
        This is a demonstration application using synthetic data for educational purposes.
        For production use, integrate with real-time data sources and validate with actual traffic data.
        """.format(accuracy))

        st.subheader("Contact Information")
        st.write("""
        For questions or suggestions about this application:
        - 📧 Email: ankit3032005@gmail.com
        - 🌐 Website: www.roadaccidentprediction.in
        - 📱 Phone: +91-8235542245
        """)

if __name__ == "__main__":
    main()
