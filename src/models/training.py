"""
Model training and evaluation module for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, precision_score, recall_score
)
import xgboost as xgb


from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Customer Churn Prediction Model
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        self.mlflow_run_id = None
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
    def prepare_features(self, features_df: pd.DataFrame, churn_labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training by merging with labels and encoding.
        
        Args:
            features_df (pd.DataFrame): Engineered features
            churn_labels (pd.DataFrame): Churn labels
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y) for modeling
        """
        logger.info("Preparing features for modeling...")
        
        # Debug: Print DataFrame columns and shapes
        logger.info(f"Features DataFrame shape: {features_df.shape}")
        logger.info(f"Features DataFrame columns: {list(features_df.columns)}")
        logger.info(f"Churn labels DataFrame shape: {churn_labels.shape}")
        logger.info(f"Churn labels DataFrame columns: {list(churn_labels.columns)}")
        
        # Merge features with churn labels
        final_dataset = features_df.merge(
            churn_labels[['userId', 'churn', 'explicit_churn', 'inactive_churn']], 
            on='userId', how='left'
        )
        
        # Check for missing churn labels
        missing_churn = final_dataset['churn'].isna().sum()
        if missing_churn > 0:
            logger.warning(f"Found {missing_churn} users with missing churn labels")
            final_dataset = final_dataset.dropna(subset=['churn'])
        
        # Prepare feature columns (exclude userId and churn-related columns)
        feature_cols = [col for col in final_dataset.columns 
                       if col not in ['userId', 'churn', 'explicit_churn', 'inactive_churn']]
        
        # Handle categorical variables
        df_model = final_dataset.copy()
        for col in config.CATEGORICAL_COLS:
            if col in df_model.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_model[col] = self.label_encoders[col].fit_transform(df_model[col].astype(str))
                else:
                    df_model[col] = self.label_encoders[col].transform(df_model[col].astype(str))
        
        # Prepare X and y
        X = df_model[feature_cols]
        y = df_model['churn']
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Prepared features: {X.shape}")
        logger.info(f"Churn distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled train and test features
        """
        logger.info("Scaling features...")
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Train the XGBoost model with k-fold cross-validation and MLflow tracking.
        
        Args:
            X_train (np.ndarray): Scaled training features
            y_train (pd.Series): Training labels
        """
        logger.info("Training XGBoost model with k-fold cross-validation...")
        
        # Set or create MLflow experiment
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run() as run:
            self.mlflow_run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_param("algorithm", "XGBoost")
            mlflow.log_param("random_state", config.RANDOM_STATE)
            mlflow.log_param("n_estimators", config.N_ESTIMATORS)
            mlflow.log_param("max_depth", config.MAX_DEPTH)
            mlflow.log_param("learning_rate", config.LEARNING_RATE)
            mlflow.log_param("subsample", config.SUBSAMPLE)
            mlflow.log_param("colsample_bytree", config.COLSAMPLE_BYTREE)
            mlflow.log_param("min_child_weight", config.MIN_CHILD_WEIGHT)
            mlflow.log_param("gamma", config.GAMMA)
            mlflow.log_param("reg_alpha", config.REG_ALPHA)
            mlflow.log_param("reg_lambda", config.REG_LAMBDA)
            mlflow.log_param("scale_pos_weight", config.SCALE_POS_WEIGHT)
            mlflow.log_param("max_delta_step", config.MAX_DELTA_STEP)
            mlflow.log_param("early_stopping_rounds", config.EARLY_STOPPING_ROUNDS)
            mlflow.log_param("eval_metric", config.EVAL_METRIC)
            mlflow.log_param("test_size", config.TEST_SIZE)
            mlflow.log_param("cv_folds", config.CV_FOLDS)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", X_train.shape[0])
            
            # Initialize XGBoost model with enhanced imbalanced data handling
            self.model = xgb.XGBClassifier(
                n_estimators=config.N_ESTIMATORS,
                max_depth=config.MAX_DEPTH,
                learning_rate=config.LEARNING_RATE,
                subsample=config.SUBSAMPLE,
                colsample_bytree=config.COLSAMPLE_BYTREE,
                min_child_weight=config.MIN_CHILD_WEIGHT,
                gamma=config.GAMMA,
                reg_alpha=config.REG_ALPHA,
                reg_lambda=config.REG_LAMBDA,
                
                # Imbalanced data specific parameters
                scale_pos_weight=config.SCALE_POS_WEIGHT,  # Primary parameter for imbalanced data
                max_delta_step=config.MAX_DELTA_STEP,      # Helps with imbalanced data convergence
                
                # Model configuration
                random_state=config.RANDOM_STATE,
                eval_metric=config.EVAL_METRIC,            # AUC is better for imbalanced data
                enable_categorical=False
                # Note: early_stopping_rounds removed for cross-validation compatibility
            )
            
            # Perform k-fold cross-validation
            logger.info(f"Performing {config.CV_FOLDS}-fold cross-validation...")
            cv_scores = self.perform_cross_validation(X_train, y_train)
            
            # Log cross-validation results
            mlflow.log_metric("cv_accuracy_mean", cv_scores['accuracy'].mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores['accuracy'].std())
            mlflow.log_metric("cv_roc_auc_mean", cv_scores['roc_auc'].mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores['roc_auc'].std())
            mlflow.log_metric("cv_f1_mean", cv_scores['f1'].mean())
            mlflow.log_metric("cv_f1_std", cv_scores['f1'].std())
            
            # Train final model on full training set
            logger.info("Training final model on full training set...")
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            logger.info("Model training completed!")
            
            # Print cross-validation results
            print(f"\nCross-Validation Results ({config.CV_FOLDS} folds):")
            print(f"Accuracy: {cv_scores['accuracy'].mean():.4f} (+/- {cv_scores['accuracy'].std() * 2:.4f})")
            print(f"ROC AUC: {cv_scores['roc_auc'].mean():.4f} (+/- {cv_scores['roc_auc'].std() * 2:.4f})")
            print(f"F1 Score: {cv_scores['f1'].mean():.4f} (+/- {cv_scores['f1'].std() * 2:.4f})")
    
    def perform_cross_validation(self, X: np.ndarray, y: pd.Series) -> Dict[str, np.ndarray]:
        """
        Perform k-fold cross-validation on the model.
        
        Args:
            X (np.ndarray): Features
            y (pd.Series): Target labels
            
        Returns:
            Dict[str, np.ndarray]: Cross-validation scores for different metrics
        """
        from sklearn.metrics import f1_score, make_scorer
        
        # Create stratified k-fold cross-validator
        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_RANDOM_STATE)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'f1': make_scorer(f1_score, pos_label=1)
        }
        
        cv_scores = {}
        for score_name, scorer in scoring.items():
            scores = cross_val_score(self.model, X, y, cv=skf, scoring=scorer)
            cv_scores[score_name] = scores
            logger.info(f"CV {score_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def evaluate_model(self, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_train (np.ndarray): Scaled training features
            X_test (np.ndarray): Scaled test features
            y_train (pd.Series): Training labels
            y_test (pd.Series): Test labels
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_pred_proba),
            'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
            'test_precision_minority': precision_score(y_test, y_test_pred, pos_label=1),
            'test_recall_minority': recall_score(y_test, y_test_pred, pos_label=1),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'feature_importance': self.get_feature_importance()
        }
        
        # Log metrics to MLflow if we have an active run
        if self.mlflow_run_id:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_metric("train_accuracy", metrics['train_accuracy'])
                mlflow.log_metric("test_accuracy", metrics['test_accuracy'])
                mlflow.log_metric("test_auc", metrics['test_auc'])
                mlflow.log_metric("test_precision_weighted", metrics['test_precision'])
                mlflow.log_metric("test_recall_weighted", metrics['test_recall'])
                mlflow.log_metric("test_precision_minority", metrics['test_precision_minority'])
                mlflow.log_metric("test_recall_minority", metrics['test_recall_minority'])
                mlflow.log_metric("precision", metrics['classification_report']['1']['precision'])
                mlflow.log_metric("recall", metrics['classification_report']['1']['recall'])
                mlflow.log_metric("f1_score", metrics['classification_report']['1']['f1-score'])
                
                # Log model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    registered_model_name=config.MLFLOW_MODEL_NAME
                )
                
        
        # Print results
        print(f"\nModel Performance:")
        print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test AUC: {metrics['test_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\nTop 10 Most Important Features:")
        print(metrics['feature_importance'].head(10))
        
        # Error Analysis Section
        print(f"\n{'='*50}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*50}")
        
        # Confusion Matrix Breakdown
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Negatives (TN):  {tn:4d} ({tn/total:.2%})")
        print(f"False Positives (FP): {fp:4d} ({fp/total:.2%}) - Model predicted churn, but customer didn't churn")
        print(f"False Negatives (FN): {fn:4d} ({fn/total:.2%}) - Model missed actual churn")
        print(f"True Positives (TP):  {tp:4d} ({tp/total:.2%})")
        
        # Error rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\nError Rates:")
        print(f"False Positive Rate: {false_positive_rate:.3f} ({false_positive_rate:.1%})")
        print(f"False Negative Rate: {false_negative_rate:.3f} ({false_negative_rate:.1%})")
        
        # Business Impact Analysis
        print(f"\nBusiness Impact Analysis:")
        print(f"Cost of False Positives: Unnecessary retention efforts for {fp} customers")
        print(f"Cost of False Negatives: Missed {fn} customers who actually churned")
        print(f"Model Sensitivity (Recall): {tp/(tp+fn):.3f} - Captures {tp/(tp+fn):.1%} of actual churners")
        print(f"Model Specificity: {tn/(tn+fp):.3f} - Correctly identifies {tn/(tn+fp):.1%} of non-churners")
        
        # Error Pattern Analysis
        print(f"\nError Pattern Analysis:")
        
        # Get prediction probabilities for error analysis
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Analyze False Positives (predicted churn but didn't churn)
        fp_indices = np.where((y_test_pred == 1) & (y_test == 0))[0]
        if len(fp_indices) > 0:
            fp_probabilities = y_test_pred_proba[fp_indices]
            print(f"False Positives Analysis:")
            print(f"  Count: {len(fp_indices)}")
            print(f"  Avg Prediction Probability: {fp_probabilities.mean():.3f}")
            print(f"  High Confidence FPs (>0.7): {np.sum(fp_probabilities > 0.7)}")
            print(f"  Low Confidence FPs (<0.6): {np.sum(fp_probabilities < 0.6)}")
        
        # Analyze False Negatives (missed actual churn)
        fn_indices = np.where((y_test_pred == 0) & (y_test == 1))[0]
        if len(fn_indices) > 0:
            fn_probabilities = y_test_pred_proba[fn_indices]
            print(f"False Negatives Analysis:")
            print(f"  Count: {len(fn_indices)}")
            print(f"  Avg Prediction Probability: {fn_probabilities.mean():.3f}")
            print(f"  Near-miss FNs (>0.4): {np.sum(fn_probabilities > 0.4)}")
            print(f"  Clear miss FNs (<0.3): {np.sum(fn_probabilities < 0.3)}")
        
        # Prediction confidence distribution
        print(f"\nPrediction Confidence Distribution:")
        high_conf_pos = np.sum(y_test_pred_proba > 0.7)
        medium_conf_pos = np.sum((y_test_pred_proba > 0.4) & (y_test_pred_proba <= 0.7))
        low_conf_pos = np.sum(y_test_pred_proba <= 0.4)
        
        print(f"High Confidence Predictions (>0.7): {high_conf_pos} ({high_conf_pos/len(y_test):.1%})")
        print(f"Medium Confidence Predictions (0.4-0.7): {medium_conf_pos} ({medium_conf_pos/len(y_test):.1%})")
        print(f"Low Confidence Predictions (<0.4): {low_conf_pos} ({low_conf_pos/len(y_test):.1%})")
        
        # Model calibration check
        print(f"\nModel Calibration Check:")
        prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(prob_bins)-1):
            bin_mask = (y_test_pred_proba >= prob_bins[i]) & (y_test_pred_proba < prob_bins[i+1])
            if np.sum(bin_mask) > 0:
                actual_rate = np.mean(y_test[bin_mask])
                predicted_rate = np.mean(y_test_pred_proba[bin_mask])
                count = np.sum(bin_mask)
                print(f"  Prob {prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}: {count:3d} samples, Actual: {actual_rate:.3f}, Predicted: {predicted_rate:.3f}")
        
        print(f"\n{'='*50}")
        print(f"END ERROR ANALYSIS")
        print(f"{'='*50}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained Random Forest model.
        
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if not self.is_trained or self.feature_names is None:
            raise ValueError("Model must be trained and feature names must be available")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and preprocessors to both local file and MLflow.
        
        Args:
            filepath (str): Path to save the model locally
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        # Save locally
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save preprocessing artifacts to MLflow if we have an active run
        if self.mlflow_run_id:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                # Save preprocessing artifacts
                temp_scaler_path = "temp_scaler.joblib"
                temp_encoders_path = "temp_encoders.joblib"
                
                joblib.dump(self.scaler, temp_scaler_path)
                joblib.dump(self.label_encoders, temp_encoders_path)
                
                mlflow.log_artifact(temp_scaler_path, "preprocessing")
                mlflow.log_artifact(temp_encoders_path, "preprocessing")
                
                # Clean up temp files
                import os
                os.remove(temp_scaler_path)
                os.remove(temp_encoders_path)
                
                logger.info(f"Model and artifacts saved to MLflow run {self.mlflow_run_id}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and preprocessors.
        
        Args:
            filepath (str): Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        for col in config.CATEGORICAL_COLS:
            if col in X_processed.columns and col in self.label_encoders:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed[self.feature_names])
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions on new data.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        for col in config.CATEGORICAL_COLS:
            if col in X_processed.columns and col in self.label_encoders:
                X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed[self.feature_names])
        
        # Make predictions
        return self.model.predict_proba(X_scaled)