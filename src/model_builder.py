"""
Model Builder
Trains machine learning models for invoice prediction and analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb


class ModelBuilder:
    def __init__(self, data_file='data/processed/processed_invoices.csv'):
        """Initialize model builder"""
        self.data_file = Path(data_file)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load processed data"""
        if not self.data_file.exists():
            print(f"❌ File not found: {self.data_file}")
            print("Please run data_processor.py first!")
            return False
        
        self.df = pd.read_csv(self.data_file)
        print(f"✅ Loaded {len(self.df)} records")
        return True
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n🔧 Preparing features...")
        
        # Select features for modeling
        feature_columns = [
            'amount', 'tax', 'year', 'month', 'quarter', 
            'day_of_week', 'day_of_month', 'week_of_year',
            'total_amount', 'tax_rate', 'vendor_avg_amount',
            'vendor_std_amount', 'vendor_invoice_count',
            'days_since_last_invoice', 'is_quarter_end', 'is_month_end'
        ]
        
        # Add vendor encoding
        if 'vendor' in self.df.columns:
            le = LabelEncoder()
            self.df['vendor_encoded'] = le.fit_transform(self.df['vendor'])
            self.label_encoders['vendor'] = le
            feature_columns.append('vendor_encoded')
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        # Prepare X and y
        X = self.df[available_features].copy()
        y = self.df['target'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        self.feature_names = available_features
        print(f"  - Selected {len(available_features)} features")
        print(f"  - Features: {available_features}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\n📊 Splitting data (test size: {test_size*100}%)...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Testing samples: {len(self.X_test)}")
        
        return True
    
    def train_model(self, model_type='xgboost'):
        """Train the selected model"""
        print(f"\n🤖 Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1
            )
            
        elif model_type == 'linear':
            self.model = LinearRegression()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        print("  - Model training complete!")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📈 Evaluating model performance...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'mae': mean_absolute_error(self.y_train, y_train_pred),
            'r2': r2_score(self.y_train, y_train_pred)
        }
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'mae': mean_absolute_error(self.y_test, y_test_pred),
            'r2': r2_score(self.y_test, y_test_pred)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"\nTraining Set:")
        print(f"  - RMSE: ${train_metrics['rmse']:,.2f}")
        print(f"  - MAE:  ${train_metrics['mae']:,.2f}")
        print(f"  - R²:   {train_metrics['r2']:.4f}")
        
        print(f"\nTesting Set:")
        print(f"  - RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"  - MAE:  ${test_metrics['mae']:,.2f}")
        print(f"  - R²:   {test_metrics['r2']:.4f}")
        
        print(f"\nCross-Validation R² Score:")
        print(f"  - Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'cv_scores': cv_scores,
            'predictions': {
                'y_test': self.y_test,
                'y_test_pred': y_test_pred
            }
        }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n" + "=" * 60)
            print("TOP 10 FEATURE IMPORTANCE")
            print("=" * 60)
            print(feature_importance.head(10))
            
            return feature_importance
        else:
            print("  - Feature importance not available for this model")
            return None
    
    def save_model(self, model_name='invoice_predictor'):
        """Save the trained model and preprocessors"""
        print(f"\n💾 Saving model...")
        
        # Save model
        model_path = self.model_dir / f'{model_name}_model.pkl'
        joblib.dump(self.model, model_path)
        print(f"  - Model saved: {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / f'{model_name}_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"  - Scaler saved: {scaler_path}")
        
        # Save label encoders
        if self.label_encoders:
            encoders_path = self.model_dir / f'{model_name}_encoders.pkl'
            joblib.dump(self.label_encoders, encoders_path)
            print(f"  - Encoders saved: {encoders_path}")
        
        # Save feature names
        features_path = self.model_dir / f'{model_name}_features.pkl'
        joblib.dump(self.feature_names, features_path)
        print(f"  - Features saved: {features_path}")
        
        return model_path
    
    def predict(self, new_data):
        """Make predictions on new data"""
        # Prepare features
        new_data_scaled = self.scaler.transform(new_data)
        
        # Predict
        predictions = self.model.predict(new_data_scaled)
        
        return predictions


def main():
    """Main execution function"""
    print("=" * 60)
    print("MODEL BUILDER & TRAINER")
    print("=" * 60)
    
    # Initialize builder
    builder = ModelBuilder()
    
    # Load data
    if not builder.load_data():
        return
    
    # Prepare features
    X, y = builder.prepare_features()
    
    # Split data
    builder.split_data(X, y)
    
    # Train multiple models and compare
    models_to_try = ['xgboost', 'random_forest', 'lightgbm']
    best_score = -np.inf
    best_model_type = None
    
    results = {}
    
    for model_type in models_to_try:
        print("\n" + "=" * 60)
        builder.train_model(model_type)
        metrics = builder.evaluate_model()
        results[model_type] = metrics
        
        if metrics['test']['r2'] > best_score:
            best_score = metrics['test']['r2']
            best_model_type = model_type
    
    # Train final model with best performing algorithm
    print("\n" + "=" * 60)
    print(f"🏆 Best model: {best_model_type} (R² = {best_score:.4f})")
    print("=" * 60)
    
    builder.train_model(best_model_type)
    builder.evaluate_model()
    builder.get_feature_importance()
    
    # Save the best model
    builder.save_model()
    
    print("\n✅ Model building complete!")


if __name__ == "__main__":
    main()