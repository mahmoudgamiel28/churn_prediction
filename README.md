# 🎵 Customer Churn Prediction ML Pipeline

**An end-to-end machine learning solution for predicting customer churn in music streaming services using behavioral analytics, MLflow experiment tracking, and containerized deployment.**

---

## 📋 Table of Contents

- [🏗️ Project Architecture](#️-project-architecture)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [🚨 Project Challenges & Key Decisions](#-project-challenges--key-decisions)
- [⚙️ Core Features](#️-core-features)
- [💻 Development & Usage](#-development--usage)
- [🔧 Configuration](#-configuration)
- [📊 Model Performance](#-model-performance)
- [🌟 Future Enhancements](#-future-enhancements)
- [📦 Dependencies](#-dependencies)

---

## 🏗️ Project Architecture

```
├── main.py                     # Main pipeline execution script
├── run_api.py                  # FastAPI application runner
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Multi-service Docker deployment
├── .dockerignore              # Docker build context exclusions
├── .env.example               # Environment variables template
├── raw_data/                   # Raw data directory
│   └── customer_churn.json
├── models/                     # Saved models directory
├── src/                        # Source code modules
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py             # API endpoints
│   │   └── models.py           # Pydantic request/response models
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py    # Data loading and cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py      # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── training.py         # Model training and evaluation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Utility functions
└── Untitled.ipynb            # Original notebook (reference)
```

---

## 🚀 Quick Start Guide

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd churn-prediction

# 2. Install Git LFS and fetch large files
git install lfs
git lfs fetch --all
git lfs pull

# 3. Build Docker images
docker build .

# 4. Start all services
docker-compose up -d

# 5. Train initial model
docker-compose --profile training up train-model

# 6. Verify deployment
curl http://localhost:8000/health
```

**Access Points**:

- 📚 **API Documentation**: http://localhost:8000/docs
- 💊 **Health Monitoring**: http://localhost:8000/health
- 🔬 **MLflow Dashboard**: http://localhost:5050

### 📡 API Usage Examples

```bash
# Health status check
curl http://localhost:8000/health

# Batch prediction from test dataset
curl -X POST 'http://localhost:8000/predict_from_test_data?file_path=customer_churn_mini.json' \
     -H 'Content-Type: application/json'
```

---

## 🚨 Project Challenges & Key Decisions

### 🎯 Business Domain Understanding

**The Challenge**: Limited knowledge of music streaming business patterns and user behavior dynamics.

**Key Uncertainties Addressed**:

- **Data Timeframe**: Unknown collection period required assumptions about seasonality and user lifecycle patterns
- **Prediction Cadence**: Business impact varies significantly between weekly, bi-weekly, or monthly prediction cycles
- **Intervention Strategy**: Model provides risk scores, but retention strategies need domain expertise

**Business Impact**: Recommendations include stakeholder collaboration to determine optimal prediction frequency based on intervention capabilities and cost-benefit analysis.

### ⏰ Churn Definition & Detection Strategy

**The Challenge**: Defining customer churn without explicit cancellation events.

**Current Implementation**:

- **30-Day Inactivity Threshold**: Users inactive for 30+ days classified as churned
- **Explicit Cancellation Signals**: Users visiting "Cancel" or "Cancellation Confirmation" pages labeled as churned

**Potential Optimizations**:

- **Reduced Time Windows**: 14-21 day thresholds may be more appropriate for high-engagement music streaming
- **Segmented Definitions**: Different inactivity periods for free vs. premium users
- **Risk Scoring**: Graduated churn probability instead of binary classification

### 🛡️ Data Leakage Prevention

**The Challenge**: Identifying features that directly indicate churn outcomes.

**Mitigation Strategy**:

```python
# Features excluded to prevent data leakage
EXCLUDED_PAGES = [
    'Cancel',
    'Cancellation Confirmation'
]
```

**Implementation**: Converted explicit churn indicators into labels rather than predictive features, ensuring temporal validity of all feature calculations.

### 📊 Technical & Data Quality Challenges

**Data Integrity Issues**:

- Missing demographic information and user agent data
- Inconsistent timestamp formats and extreme outlier values
- Session duration approximation limitations (only measures event span, not final song completion)

**Feature Engineering Decisions**:

- Skip detection threshold set at 80% song completion (configurable)
- Completion rates capped at 10.0 to handle data anomalies
- Device type extraction from user agent strings with fallback handling

---

## ⚙️ Core Features

### 🧹 Data Preprocessing Pipeline

- **Multi-format Data Loading**: JSON ingestion with robust error handling
- **Data Quality Assurance**: Missing value imputation and outlier detection
- **Temporal Processing**: Timestamp standardization and chronological validation
- **Churn Label Creation**: Combined explicit cancellation signals with inactivity-based detection

### 🔧 Comprehensive Feature Engineering

#### 📱 Session Analytics

- **Session Count**: Total unique user sessions
- **Engagement Depth**: Average songs per session
- **Session Duration**: Time-based activity patterns
- **Activity Consistency**: Ratio of active days to total activity span

#### 👥 User Demographics

- **Gender Encoding**: Binary demographic classification
- **Subscription Status**: Premium vs. free usage ratio
- **Device Preference**: Platform-based usage patterns (Mobile/Desktop/Tablet)

#### 🎵 Behavioral Event Tracking

Comprehensive user action analytics across 15+ event types:

- **Music Interaction**: Song plays, skips, playlist additions
- **Engagement Actions**: Thumbs up/down, social interactions
- **Platform Navigation**: Help visits, settings changes, subscription modifications
- **Technical Events**: Error occurrences, logout frequency

#### 🎶 Music Consumption Analysis

- **Skip Behavior**: Skip frequency and completion rate patterns
- **Listening Quality**: Song completion consistency metrics
- **Engagement Depth**: Total consumption and behavioral trends

### 🤖 Machine Learning & MLflow Integration

- **Algorithm**: Logistic Regression with comprehensive preprocessing
- **Data Handling**: Stratified train/test splits with feature scaling
- **Evaluation Suite**: Multi-metric performance assessment (AUC, precision, recall, F1)
- **Experiment Tracking**: MLflow integration for parameter logging, metric tracking, and model versioning
- **Model Registry**: Centralized model management with deployment capabilities

### 🌐 Production-Ready API & Deployment

- **FastAPI Framework**: High-performance REST API with automatic documentation
- **Docker Containerization**: Multi-service deployment with health monitoring
- **MLflow Integration**: Model serving and experiment visualization
- **Monitoring**: Health checks and service status endpoints


## 💻 Development & Usage

### 🏃‍♂️ Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py

# Start MLflow tracking server
mlflow ui --port 5050

# Launch API server
python run_api.py
```

### 🧱 Modular Usage

```python
# Import pipeline components
from src.data.preprocessing import preprocess_data
from src.features.engineering import engineer_features
from src.models.training import ChurnPredictor

# Execute pipeline steps
cleaned_data, churn_labels = preprocess_data()
user_features = engineer_features(cleaned_data)

# Train and evaluate model
predictor = ChurnPredictor()
X, y = predictor.prepare_features(user_features, churn_labels)
model_metrics = predictor.train_and_evaluate(X, y)
```

### 🔮 Making Predictions

```python
from main import predict_churn
import pandas as pd

# Load feature data
features_df = pd.read_csv('user_features.csv')

# Generate predictions
predictions, probabilities = predict_churn('models/churn_predictor.joblib', features_df)
```

---

## 🔧 Configuration

### ⚙️ Configuration Management

Modify `config.py` to customize:

- **Data Sources**: Input file paths and formats
- **Model Parameters**: Algorithm hyperparameters and training settings
- **Feature Engineering**: Skip thresholds, session definitions, event types
- **Output Destinations**: Model storage and logging configurations

### 🌍 Environment Variables

```bash
# Core service configuration
MLFLOW_TRACKING_URI=http://localhost:5050    # MLflow server endpoint
MODEL_DIR=./models                           # Model storage directory
API_PORT=8000                               # API service port

# Docker volume configuration
./models:/app/models                        # Persistent model storage
mlflow_data:/mlflow                         # MLflow tracking persistence
```


---

## 📊 Model Performance

### 📈 Evaluation Metrics

The pipeline provides comprehensive model assessment:

- **Classification Accuracy**: Training and validation performance
- **ROC-AUC Score**: Area under the receiver operating characteristic curve
- **Precision & Recall**: Class-specific performance metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: Individual feature contribution analysis

### 🎯 Model Architecture

- **Algorithm**: Logistic Regression with L2 regularization
- **Preprocessing**: StandardScaler for numerical features, categorical encoding
- **Validation**: Stratified train/test split for balanced evaluation
- **Tracking**: MLflow experiment logging with parameter and metric persistence

---

## 🌟 Future Enhancements

### 🎯 Model Performance & Algorithm Improvements

**Hyperparameter Optimization**:

- **Grid Search/Random Search**: Systematically tune model parameters for optimal performance
  - Logistic Regression: Regularization strength (C), penalty type (L1/L2), solver optimization
  - Random Forest: Number of trees, maximum depth, minimum samples for splitting
  - XGBoost: Learning rate, tree depth, number of estimators, subsample ratio
- **Bayesian Search**: Efficient hyperparameter optimization using Bayesian optimization techniques (e.g., scikit-optimize, Optuna) for intelligent parameter space exploration with fewer iterations

**Advanced Machine Learning Techniques**:

- **Feature Selection**: Apply Recursive Feature Elimination (RFE), SelectKBest, or LASSO regularization to identify most predictive features
- **Ensemble Methods**: Combine multiple algorithms using Voting Classifiers, Stacking, or Bagging for improved accuracy
- **Algorithm Experimentation**: Test gradient boosting methods like XGBoost, LightGBM, and CatBoost for potentially better performance

### 📊 Churn Definition & Business Logic Refinements

**Optimized Churn Detection Windows**:

- **Variable Time Periods**: Test and compare different inactivity thresholds (7, 14, 21, 30 days) to find optimal detection timing
- **User Segmentation**: Apply different churn definitions based on user types (new vs. established customers, free vs. paid users)
- **Seasonal Considerations**: Adjust churn detection logic to account for natural usage fluctuations during holidays and seasonal periods

**Enhanced Churn Classification**:

- **Probabilistic Scoring**: Replace binary churn labels with probability scores for more nuanced risk assessment
- **Behavioral Segmentation**: Create distinct user segments with tailored churn definitions based on usage patterns
- **Cohort Analysis**: Track and analyze churn patterns across different user acquisition periods

### 🔧 Feature Engineering & Data Enhancement

**Temporal & Behavioral Features**:

- **Recent Activity Trends**: Compare last 7-day activity against historical averages to detect engagement decline
- **Time-based Patterns**: Capture weekend vs. weekday usage, peak listening hours, and session timing preferences
- **Engagement Trajectory**: Calculate rate of change in user engagement over time

**Advanced User Behavior Analysis**:

- **Music Preference Features**: Analyze genre diversity, artist loyalty, and music discovery patterns
- **Social Interaction Metrics**: Track friend additions, playlist sharing, and community engagement
- **Support & Payment History**: Include customer service interactions and subscription change patterns
- **Feature Interactions**: Create composite features combining multiple behavioral indicators (e.g., skip_rate × session_frequency)

### 🛡️ Data Quality & Validation Improvements

**Automated Data Integrity**:

- **Quality Assurance Checks**: Implement automated validation for data completeness, consistency, and anomaly detection
- **Feature Audit System**: Regular review of feature importance to identify and prevent new sources of data leakage
- **Temporal Validation**: Ensure all predictive features are derived exclusively from historical data preceding churn events

**Enhanced Data Sources**:

- **User Feedback Integration**: Collect explicit churn reasons through surveys and satisfaction measurements
- **Extended Behavioral Tracking**: Monitor playlist creation, music discovery, and social sharing activities
- **External Market Data**: Incorporate industry trends, competitor analysis, and economic indicators that may influence churn

### 📈 Business Impact & Operational Integration

**Intervention & Campaign Optimization**:

- **Retention Campaign Tracking**: Monitor and measure success rates of different customer retention strategies
- **Revenue Impact Analysis**: Connect churn predictions to customer lifetime value and revenue implications
- **Cost-Benefit Optimization**: Balance prediction accuracy with intervention costs for maximum business impact

---

## 📦 Dependencies

### 🔧 Core Technology Stack

| Component        | Version  | Purpose                                  |
| ---------------- | -------- | ---------------------------------------- |
| **pandas**       | ≥2.0.0   | Data manipulation and analysis           |
| **scikit-learn** | ≥1.3.0   | Machine learning algorithms              |
| **fastapi**      | ≥0.100.0 | High-performance API framework           |
| **mlflow**       | ≥2.0.0   | Experiment tracking and model management |
| **uvicorn**      | Latest   | ASGI server for FastAPI                  |
| **joblib**       | Latest   | Model serialization and persistence      |

### 🐳 Deployment Infrastructure

- **Docker & Docker Compose**: Containerization and multi-service orchestration
- **Python 3.8+**: Runtime environment with ML ecosystem support

**Complete Dependency List**: See `requirements.txt` for detailed version specifications and additional packages.

---

<div align="center">

_Leveraging data science to enhance user experience and business growth_

</div>
