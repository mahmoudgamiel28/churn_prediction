# Customer Churn Prediction ML Pipeline

A modular machine learning pipeline for predicting customer churn using user activity data with MLflow experiment tracking and Docker deployment.

## Project Structure

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

## Features

### Data Preprocessing

- Load data from JSON format
- Clean invalid entries and handle missing values
- Convert timestamps and handle data types
- Create churn labels (explicit + inactive churn)

### Feature Engineering

- **Session Features**: Number of sessions, average session duration, days active
- **Demographic Features**: Gender, subscription level, device type
- **Event Features**: Counts of user actions (NextSong, Logout, Thumbs Up/Down, etc.)
- **Skip Features**: Song skip behavior analysis

### Model Training & MLflow Integration

- Logistic Regression with proper preprocessing
- Train/test split with stratification
- Feature scaling with StandardScaler
- Categorical encoding
- Comprehensive model evaluation
- **MLflow experiment tracking** with metrics, parameters, and model registry
- **Model versioning** and deployment from MLflow registry

### API & Deployment

- **FastAPI REST API** for predictions and monitoring
- **Docker containerization** with multi-service deployment
- **Health checks** and service monitoring
- **MLflow UI** for experiment visualization

## Usage

### 🐳 Docker Deployment (Recommended)

#### Quick Start with Docker Compose

```bash
# Start all services (MLflow + API)
docker-compose up -d

# Train model (one-time)
docker-compose --profile training up train-model

# View logs
docker-compose logs -f churn-api
docker-compose logs -f mlflow
```

#### Access Services

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000

#### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Batch prediction from test data (pass filename as query parameter)
curl --location --request POST 'http://localhost:8000/predict_from_test_data?file_path=customer_churn_mini.json' \
--header 'Content-Type: application/json'
```

### 🖥️ Local Development

#### Run Complete Pipeline

```bash
python main.py
```

#### Start MLflow Server

```bash
mlflow ui --port 5000
```

#### Start API Server

```bash
python run_api.py
```

### Use Individual Modules

```python
from src.data.preprocessing import preprocess_data
from src.features.engineering import engineer_features
from src.models.training import ChurnPredictor

# Load and preprocess data
cleaned_data, churn_labels = preprocess_data()

# Engineer features
user_features = engineer_features(cleaned_data)

# Train model
predictor = ChurnPredictor()
X, y = predictor.prepare_features(user_features, churn_labels)
# ... continue with training
```

### Make Predictions

```python
from main import predict_churn
import pandas as pd

# Load your feature data
features_df = pd.read_csv('new_user_features.csv')

# Make predictions
predictions, probabilities = predict_churn('models/churn_predictor.joblib', features_df)
```

## Configuration

Modify `config.py` to adjust:

- Data paths
- Model hyperparameters
- Feature engineering parameters
- Required page types for event counting

## Key Features of This Implementation

### ML Best Practices

1. **Modular Design**: Separate modules for data, features, and models
2. **Configuration Management**: Centralized config for easy parameter tuning
3. **Logging**: Comprehensive logging throughout the pipeline
4. **Error Handling**: Proper exception handling and validation
5. **Reproducibility**: Fixed random seeds and versioned preprocessing
6. **Model Persistence**: Save and load trained models

### Code Quality

1. **Type Hints**: Function signatures with proper typing
2. **Documentation**: Detailed docstrings for all functions
3. **Separation of Concerns**: Each module has a single responsibility
4. **Reusability**: Functions can be used independently
5. **Testability**: Modular design enables easy unit testing

### Scalability

1. **Class-based Model**: Encapsulated model training and prediction
2. **Pipeline Design**: Easy to add new features or models
3. **Config-driven**: Change behavior without code changes
4. **Extensible**: Easy to add new feature types or models

## Model Performance

The pipeline trains a Logistic Regression model and provides:

- Training and test accuracy
- AUC score
- Classification report
- Confusion matrix
- Feature importance analysis

## 🚀 Deployment Options

### Production Deployment

#### Docker Build

```bash
# Build custom image
docker build -t churn-prediction-api .

# Run with custom configuration
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 \
  -v $(pwd)/models:/app/models \
  churn-prediction-api
```

#### Environment Configuration

```bash
# Copy and modify environment template
cp .env.example .env
# Edit .env with your configuration
```

### Scaling with Docker Swarm/Kubernetes

The application is container-ready for orchestration platforms:

- **Health checks** for container management
- **Environment-based configuration**
- **Volume mounts** for persistent data
- **Service dependencies** properly defined

## 🔧 Configuration

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MODEL_DIR`: Model storage directory
- `API_PORT`: API server port (default: 8000)

### Docker Volumes

- `./models`: Trained model storage
- `mlflow_data`: MLflow tracking data persistence

## 📊 Monitoring & Observability

### MLflow Integration

- **Experiment tracking**: Parameters, metrics, artifacts
- **Model registry**: Version management and staging
- **Model comparison**: Compare different training runs
- **Artifact storage**: Models, preprocessors, reports

## Dependencies

### Core Dependencies

- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning
- `fastapi>=0.100.0` - API framework
- `mlflow>=2.0.0` - Experiment tracking

See `requirements.txt` for the complete list of dependencies.
