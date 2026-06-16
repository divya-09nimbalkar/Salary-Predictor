"""
Employee Salary Predictor - Production-Grade ML Model
Predicts employee salaries based on features like experience, education, role, etc.
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('salary_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SalaryPredictor:
    """Production-grade salary prediction model."""
    
    def __init__(self, model_path='salary_model.joblib', scaler_path='salary_scaler.joblib'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        
    def _generate_synthetic_data(self, n_samples=500):
        """Generate realistic synthetic salary data for demonstration."""
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        data = {
            'years_experience': np.random.randint(0, 40, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'job_role': np.random.choice(['Analyst', 'Developer', 'Manager', 'Senior Manager', 'Director'], n_samples),
            'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Finance', 'Operations'], n_samples),
            'performance_score': np.random.uniform(2.0, 5.0, n_samples),
            'certifications_count': np.random.randint(0, 10, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create salary target with realistic relationship
        base_salary = 30000
        exp_factor = df['years_experience'] * 1500
        edu_factor = df['education_level'].map({'High School': 0, 'Bachelor': 15000, 'Master': 30000, 'PhD': 45000})
        role_factor = df['job_role'].map({'Analyst': 5000, 'Developer': 20000, 'Manager': 35000, 'Senior Manager': 50000, 'Director': 75000})
        perf_factor = df['performance_score'] * 3000
        cert_factor = df['certifications_count'] * 1000
        noise = np.random.normal(0, 5000, n_samples)
        
        df['salary'] = base_salary + exp_factor + edu_factor + role_factor + perf_factor + cert_factor + noise
        df['salary'] = df['salary'].clip(lower=25000)  # Minimum salary floor
        
        logger.info(f"Generated data shape: {df.shape}")
        return df
    
    def preprocess_data(self, df, fit_encoders=False):
        """Preprocess and encode features."""
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include='object').columns
        
        for col in categorical_cols:
            if fit_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                logger.info(f"Fitted encoder for {col}")
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def train(self, df=None):
        """Train the salary prediction model."""
        logger.info("=" * 50)
        logger.info("Starting model training...")
        
        # Generate or use provided data
        if df is None:
            df = self._generate_synthetic_data()
        else:
            logger.info(f"Using provided data with shape {df.shape}")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, fit_encoders=True)
        
        # Separate features and target
        X = df_processed.drop('salary', axis=1)
        y = df_processed['salary']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create pipeline with scaling and model
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        logger.info("Training Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        logger.info(f"Train R² Score: {train_r2:.4f}")
        logger.info(f"Test R² Score: {test_r2:.4f}")
        logger.info(f"Test RMSE: ${test_rmse:.2f}")
        logger.info(f"Test MAE: ${test_mae:.2f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        logger.info("Top 3 Important Features:")
        for idx, row in feature_importance_df.head(3).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        logger.info("Model training completed successfully!")
        logger.info("=" * 50)
        
        return self.model, self.scaler
    
    def predict(self, features_dict):
        """Predict salary for given employee features."""
        if not self.is_trained or self.model is None:
            logger.error("Model not trained. Train model first.")
            raise ValueError("Model not trained")
        
        try:
            # Convert to DataFrame
            df_input = pd.DataFrame([features_dict])
            
            # Preprocess (encode categoricals)
            df_processed = self.preprocess_data(df_input, fit_encoders=False)
            
            # Scale
            X_scaled = self.scaler.transform(df_processed)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            logger.info(f"Prediction for {features_dict}: ${prediction:.2f}")
            
            return prediction
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def save_model(self):
        """Save trained model and scaler."""
        if self.model is None:
            logger.error("No model to save. Train first.")
            return
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoders, 'label_encoders.joblib')
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model and scaler."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoders = joblib.load('label_encoders.joblib')
        self.is_trained = True
        logger.info("Model loaded successfully!")
        return True


def main():
    """Main execution function."""
    logger.info("Employee Salary Predictor - Starting Application")
    
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Train model
    predictor.train()
    predictor.save_model()
    
    # Test predictions
    logger.info("\n" + "=" * 50)
    logger.info("Testing predictions with sample employees:")
    logger.info("=" * 50)
    
    test_cases = [
        {'years_experience': 5, 'education_level': 'Bachelor', 'job_role': 'Developer', 
         'department': 'Engineering', 'performance_score': 4.2, 'certifications_count': 2},
        {'years_experience': 15, 'education_level': 'Master', 'job_role': 'Manager', 
         'department': 'Engineering', 'performance_score': 4.8, 'certifications_count': 5},
        {'years_experience': 20, 'education_level': 'PhD', 'job_role': 'Director', 
         'department': 'Finance', 'performance_score': 4.9, 'certifications_count': 8},
    ]
    
    for i, employee in enumerate(test_cases, 1):
        salary = predictor.predict(employee)
        logger.info(f"Employee {i} - Predicted Salary: ${salary:,.2f}")
    
    logger.info("=" * 50)
    logger.info("Application completed successfully!")


if __name__ == "__main__":
    main()
