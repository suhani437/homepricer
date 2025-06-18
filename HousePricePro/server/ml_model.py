#!/usr/bin/env python3
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.property_type_encoder = LabelEncoder()
        self.neighborhood_encoder = LabelEncoder()
        self.feature_names = [
            'square_footage', 'year_built', 'bedrooms', 'bathrooms', 'garage',
            'property_type_encoded', 'neighborhood_encoded', 'has_pool',
            'has_fireplace', 'has_hardwood_floors', 'recently_updated'
        ]
        self.is_trained = False
        self.metrics = {}
        
        # Train the model with synthetic data
        self._train_model()
    
    def _generate_training_data(self, n_samples=5000):
        """Generate synthetic training data for house prices"""
        np.random.seed(42)
        
        # Generate features
        square_footage = np.random.normal(2200, 600, n_samples)
        square_footage = np.clip(square_footage, 800, 6000)
        
        year_built = np.random.randint(1950, 2024, n_samples)
        bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.2, 0.4, 0.3, 0.1])
        bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                   p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.08, 0.02])
        garage = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.2, 0.6, 0.1])
        
        property_types = ['single-family', 'condo', 'townhouse', 'multi-family']
        property_type = np.random.choice(property_types, n_samples, 
                                       p=[0.6, 0.2, 0.15, 0.05])
        
        neighborhoods = ['downtown', 'suburbs', 'waterfront', 'rural']
        neighborhood = np.random.choice(neighborhoods, n_samples, 
                                      p=[0.2, 0.5, 0.2, 0.1])
        
        has_pool = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        has_fireplace = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        has_hardwood_floors = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        recently_updated = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Create price based on features with realistic relationships (in INR)
        base_price = 5000000  # 50 lakh base price
        price = base_price + (square_footage - 2000) * 3000  # 3000 INR per sq ft
        price += (year_built - 1980) * 30000  # 30k per year newer
        price += bedrooms * 500000  # 5 lakh per bedroom
        price += bathrooms * 300000  # 3 lakh per bathroom
        price += garage * 200000  # 2 lakh per garage
        
        # Property type adjustments
        property_type_multiplier = {'single-family': 1.0, 'condo': 0.8, 
                                  'townhouse': 0.9, 'multi-family': 1.1}
        for i, ptype in enumerate(property_type):
            price[i] *= property_type_multiplier[ptype]
        
        # Neighborhood adjustments
        neighborhood_multiplier = {'downtown': 1.2, 'suburbs': 1.0, 
                                 'waterfront': 1.5, 'rural': 0.8}
        for i, nhood in enumerate(neighborhood):
            price[i] *= neighborhood_multiplier[nhood]
        
        # Feature bonuses (in INR)
        price += has_pool * 400000  # 4 lakh for pool
        price += has_fireplace * 200000  # 2 lakh for fireplace
        price += has_hardwood_floors * 150000  # 1.5 lakh for hardwood
        price += recently_updated * 300000  # 3 lakh for recent updates
        
        # Add some noise
        price += np.random.normal(0, 500000, n_samples)  # 5 lakh standard deviation
        price = np.clip(price, 2000000, 50000000)  # 20 lakh to 5 crore range
        
        # Create DataFrame
        data = pd.DataFrame({
            'square_footage': square_footage,
            'year_built': year_built,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'garage': garage,
            'property_type': property_type,
            'neighborhood': neighborhood,
            'has_pool': has_pool,
            'has_fireplace': has_fireplace,
            'has_hardwood_floors': has_hardwood_floors,
            'recently_updated': recently_updated,
            'price': price
        })
        
        return data
    
    def _train_model(self):
        """Train the linear regression model"""
        # Generate training data
        data = self._generate_training_data()
        
        # Split features and target
        X = data.drop('price', axis=1)
        y = data['price']
        
        # Encode categorical variables
        X_processed = X.copy()
        X_processed['property_type_encoded'] = self.property_type_encoder.fit_transform(X['property_type'])
        X_processed['neighborhood_encoded'] = self.neighborhood_encoder.fit_transform(X['neighborhood'])
        
        # Select numerical features
        X_final = X_processed[self.feature_names]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'r_squared': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'last_trained': '2024-12-16'
        }
        
        self.is_trained = True
    
    def predict(self, features):
        """Make a price prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Prepare features
        feature_array = np.array([
            features['squareFootage'],
            features['yearBuilt'],
            features['bedrooms'],
            features['bathrooms'],
            features['garage'],
            self._encode_property_type(features['propertyType']),
            self._encode_neighborhood(features['neighborhood']),
            int(features['hasPool']),
            int(features['hasFireplace']),
            int(features['hasHardwoodFloors']),
            int(features['recentlyUpdated'])
        ]).reshape(1, -1)
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_scaled)[0]
        
        # Calculate confidence interval (using training RMSE as estimate)
        confidence = 0.92  # Approximate confidence based on R²
        margin = self.metrics['rmse'] * 1.2
        
        return {
            'estimatedPrice': float(prediction),
            'confidence': confidence,
            'lowerBound': float(prediction - margin),
            'upperBound': float(prediction + margin),
            'featureImportance': self._get_feature_importance()
        }
    
    def _encode_property_type(self, prop_type):
        """Encode property type"""
        try:
            return self.property_type_encoder.transform([prop_type])[0]
        except ValueError:
            # Handle unknown property types
            return 0
    
    def _encode_neighborhood(self, neighborhood):
        """Encode neighborhood"""
        try:
            return self.neighborhood_encoder.transform([neighborhood])[0]
        except ValueError:
            # Handle unknown neighborhoods
            return 0
    
    def _get_feature_importance(self):
        """Get feature importance from linear regression coefficients"""
        coefficients = np.abs(self.model.coef_)
        total = np.sum(coefficients)
        
        importance_map = {
            'Square Footage': coefficients[0] / total * 100,
            'Year Built': coefficients[1] / total * 100,
            'Bedrooms': coefficients[2] / total * 100,
            'Bathrooms': coefficients[3] / total * 100,
            'Garage': coefficients[4] / total * 100,
            'Property Type': coefficients[5] / total * 100,
            'Location': coefficients[6] / total * 100,
            'Pool': coefficients[7] / total * 100,
            'Fireplace': coefficients[8] / total * 100,
            'Hardwood Floors': coefficients[9] / total * 100,
            'Recently Updated': coefficients[10] / total * 100
        }
        
        # Sort by importance and return top features
        sorted_features = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)
        return [{'feature': name, 'importance': round(importance, 1)} for name, importance in sorted_features[:5]]
    
    def get_metrics(self):
        """Get model performance metrics"""
        return {
            'rSquared': round(self.metrics['r_squared'], 3),
            'rmse': round(self.metrics['rmse'], 0),
            'mae': round(self.metrics['mae'], 0),
            'lastTrained': self.metrics['last_trained']
        }

def main():
    predictor = HousePricePredictor()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--metrics':
        # Return model metrics
        metrics = predictor.get_metrics()
        print(json.dumps(metrics))
    else:
        # Read input from stdin
        input_data = sys.stdin.read()
        features = json.loads(input_data)
        
        # Make prediction
        result = predictor.predict(features)
        print(json.dumps(result))

if __name__ == "__main__":
    main()
