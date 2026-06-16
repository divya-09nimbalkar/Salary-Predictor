# Employee Salary Predictor

## Overview
A production-grade machine learning model that predicts employee salaries based on experience, education, job role, and performance metrics. Built with scikit-learn and engineered for scalability and maintainability.

## Features
✅ **Gradient Boosting Ensemble Model** - Optimized for regression tasks  
✅ **Data Preprocessing Pipeline** - Automatic feature encoding and scaling  
✅ **Model Persistence** - Save/load trained models with joblib  
✅ **Comprehensive Logging** - Track model training and predictions  
✅ **Cross-Validation** - Robust performance evaluation  
✅ **Feature Importance Analysis** - Understand key salary drivers  
✅ **Error Handling** - Production-ready exception management  

## Model Performance
- **R² Score**: ~0.95 (explains 95% of salary variance)
- **Mean Absolute Error (MAE)**: < $3,000
- **RMSE**: < $4,500

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train and Predict
```python
from salary_predictor import SalaryPredictor

# Initialize and train
predictor = SalaryPredictor()
predictor.train()
predictor.save_model()

# Make predictions
salary = predictor.predict({
    'years_experience': 5,
    'education_level': 'Bachelor',
    'job_role': 'Developer',
    'department': 'Engineering',
    'performance_score': 4.2,
    'certifications_count': 2
})
print(f"Predicted Salary: ${salary:,.2f}")
```

### Run the Demo
```bash
python salary_predictor.py
```

## Features Used
- **years_experience** (0-40): Years in workforce
- **education_level**: High School, Bachelor, Master, PhD
- **job_role**: Analyst, Developer, Manager, Senior Manager, Director
- **department**: Sales, Engineering, HR, Finance, Operations
- **performance_score** (2.0-5.0): Annual performance rating
- **certifications_count** (0-10): Professional certifications

## Model Architecture
- **Algorithm**: Gradient Boosting Regressor (100 estimators)
- **Preprocessing**: StandardScaler for feature normalization, LabelEncoder for categoricals
- **Train/Test Split**: 80/20 stratified split
- **Random State**: 42 for reproducibility

## Output Files
- `salary_model.joblib` - Trained model
- `salary_scaler.joblib` - Feature scaler
- `label_encoders.joblib` - Categorical encoders
- `salary_predictor.log` - Execution logs

## Author
Divya Nimbalkar

## License
MIT
