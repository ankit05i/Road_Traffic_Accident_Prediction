"""
Machine Learning Models for Indian Traffic Accident Prediction
Implements various ML algorithms for accident prediction and severity estimation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccidentPredictionModel:
    """Machine Learning models for traffic accident prediction"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'severity') -> tuple:
        """Prepare features for machine learning"""

        # Define feature columns
        numeric_features = [
            'temperature', 'humidity', 'rainfall', 'wind_speed', 'visibility',
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'is_peak_hour'
        ]

        categorical_features = [
            'state', 'road_type', 'collision_type', 'vehicle_type',
            'traffic_violation', 'weather_condition', 'junction_type', 'area_type'
        ]

        # Create feature matrix
        X = df[numeric_features + categorical_features].copy()

        # Handle missing values
        X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
        X[categorical_features] = X[categorical_features].fillna('Unknown')

        # Encode categorical variables
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))

        # Prepare target variable
        if target_col == 'severity':
            # Multi-class classification
            y = df[target_col].copy()
            if 'severity_encoder' not in self.encoders:
                self.encoders['severity_encoder'] = LabelEncoder()
                y = self.encoders['severity_encoder'].fit_transform(y)
            else:
                y = self.encoders['severity_encoder'].transform(y)
        else:
            # Binary classification or regression
            y = df[target_col].copy()

        # Scale numeric features
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X[numeric_features] = self.scalers['scaler'].fit_transform(X[numeric_features])
        else:
            X[numeric_features] = self.scalers['scaler'].transform(X[numeric_features])

        logger.info(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
        return X, y

    def train_random_forest(self, X_train, y_train, X_test, y_test) -> dict:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")

        rf_model = RandomForestClassifier(
            n_estimators=MODEL_CONFIG['N_ESTIMATORS'],
            max_depth=MODEL_CONFIG['MAX_DEPTH'],
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # Predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)

        performance = {
            'model_name': 'Random Forest',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }

        self.models['random_forest'] = rf_model
        self.model_performance['random_forest'] = performance
        self.feature_importance['random_forest'] = rf_model.feature_importances_

        logger.info(f"Random Forest Accuracy: {accuracy:.4f}")
        return performance

    def train_xgboost(self, X_train, y_train, X_test, y_test) -> dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")

        xgb_model = xgb.XGBClassifier(
            n_estimators=MODEL_CONFIG['N_ESTIMATORS'],
            max_depth=MODEL_CONFIG['MAX_DEPTH'],
            learning_rate=MODEL_CONFIG['LEARNING_RATE'],
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            eval_metric='mlogloss'
        )

        xgb_model.fit(X_train, y_train)

        # Predictions
        y_pred = xgb_model.predict(X_test)

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)

        performance = {
            'model_name': 'XGBoost',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }

        self.models['xgboost'] = xgb_model
        self.model_performance['xgboost'] = performance
        self.feature_importance['xgboost'] = xgb_model.feature_importances_

        logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
        return performance

    def train_logistic_regression(self, X_train, y_train, X_test, y_test) -> dict:
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression model...")

        lr_model = LogisticRegression(
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            max_iter=1000,
            multi_class='ovr'
        )

        lr_model.fit(X_train, y_train)

        # Predictions
        y_pred = lr_model.predict(X_test)

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)

        performance = {
            'model_name': 'Logistic Regression',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }

        self.models['logistic_regression'] = lr_model
        self.model_performance['logistic_regression'] = performance

        logger.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
        return performance

    def train_lightgbm(self, X_train, y_train, X_test, y_test) -> dict:
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")

        lgb_model = lgb.LGBMClassifier(
            n_estimators=MODEL_CONFIG['N_ESTIMATORS'],
            max_depth=MODEL_CONFIG['MAX_DEPTH'],
            learning_rate=MODEL_CONFIG['LEARNING_RATE'],
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            verbose=-1
        )

        lgb_model.fit(X_train, y_train)

        # Predictions
        y_pred = lgb_model.predict(X_test)

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5)

        performance = {
            'model_name': 'LightGBM',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }

        self.models['lightgbm'] = lgb_model
        self.model_performance['lightgbm'] = performance
        self.feature_importance['lightgbm'] = lgb_model.feature_importances_

        logger.info(f"LightGBM Accuracy: {accuracy:.4f}")
        return performance

    def train_all_models(self, df: pd.DataFrame, target_col: str = 'severity'):
        """Train all models and compare performance"""
        logger.info("Preparing data for model training...")

        # Prepare features
        X, y = self.prepare_features(df, target_col)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['TEST_SIZE'],
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            stratify=y
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train all models
        models_to_train = [
            self.train_random_forest,
            self.train_xgboost,
            self.train_logistic_regression,
            self.train_lightgbm
        ]

        for train_func in models_to_train:
            try:
                train_func(X_train, y_train, X_test, y_test)
            except Exception as e:
                logger.error(f"Error training model: {e}")

        # Compare performance
        self.compare_models()

        return X_test, y_test

    def compare_models(self):
        """Compare performance of all trained models"""
        logger.info("\n=== MODEL PERFORMANCE COMPARISON ===")

        performance_df = pd.DataFrame([
            {
                'Model': perf['model_name'],
                'Accuracy': perf['accuracy'],
                'CV Mean': perf['cv_mean'],
                'CV Std': perf['cv_std']
            }
            for perf in self.model_performance.values()
        ])

        performance_df = performance_df.sort_values('Accuracy', ascending=False)
        print(performance_df.to_string(index=False))

        # Find best model
        best_model_name = performance_df.iloc[0]['Model']
        logger.info(f"\nBest performing model: {best_model_name}")

        return performance_df

    def predict_accident_risk(self, input_data: dict) -> dict:
        """Predict accident risk for given conditions"""
        if not self.models:
            logger.error("No trained models available. Please train models first.")
            return {}

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Prepare features (without target)
        try:
            X, _ = self.prepare_features(input_df, target_col='severity')
        except:
            # Handle case where severity column doesn't exist
            numeric_features = [
                'temperature', 'humidity', 'rainfall', 'wind_speed', 'visibility',
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'is_peak_hour'
            ]
            categorical_features = [
                'state', 'road_type', 'collision_type', 'vehicle_type',
                'traffic_violation', 'weather_condition', 'junction_type', 'area_type'
            ]

            X = input_df[numeric_features + categorical_features].copy()

            # Fill missing values and encode
            X[numeric_features] = X[numeric_features].fillna(0)
            X[categorical_features] = X[categorical_features].fillna('Unknown')

            for col in categorical_features:
                if col in self.encoders:
                    try:
                        X[col] = self.encoders[col].transform(X[col].astype(str))
                    except:
                        X[col] = 0  # Handle unknown categories
                else:
                    X[col] = 0

            if 'scaler' in self.scalers:
                X[numeric_features] = self.scalers['scaler'].transform(X[numeric_features])

        predictions = {}

        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                pred_proba = model.predict_proba(X)[0]

                # Decode prediction if needed
                if 'severity_encoder' in self.encoders:
                    pred_label = self.encoders['severity_encoder'].inverse_transform([pred])[0]
                else:
                    pred_label = pred

                predictions[model_name] = {
                    'prediction': pred_label,
                    'probabilities': pred_proba.tolist(),
                    'confidence': max(pred_proba)
                }
            except Exception as e:
                logger.error(f"Error making prediction with {model_name}: {e}")

        return predictions

    def save_models(self, filepath: str = None):
        """Save trained models"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"models/accident_prediction_models_{timestamp}.joblib"

        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")

        return filepath

    def load_models(self, filepath: str):
        """Load trained models"""
        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data.get('feature_importance', {})
        self.model_performance = model_data.get('model_performance', {})

        logger.info(f"Models loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from a main script
    print("Accident Prediction Model module loaded successfully!")
    print("Use this module to train and predict traffic accidents.")
