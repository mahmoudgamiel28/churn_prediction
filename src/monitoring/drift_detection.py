"""
Drift Detection Module for Customer Churn Prediction Pipeline

This module provides comprehensive drift detection capabilities including:
- Data drift detection using statistical tests
- Concept drift detection based on model performance
- Integration with Evidently for detailed drift reports
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Evidently imports
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import (
    DataDriftTable, 
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    DatasetSummaryMetric,

)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns,
    TestShareOfMissingValues,
    TestMostImportantFeaturesDrift
)

# Scipy imports for statistical tests
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class DriftDetector:
    """
    Main drift detection class that handles both data drift and concept drift detection.
    """
    
    def __init__(self, config, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the drift detector.
        
        Args:
            config: Configuration object containing model parameters
            reference_data: Reference dataset for comparison (typically training data)
        """
        self.config = config
        self.reference_data = reference_data
        self.logger = logging.getLogger(__name__)
        
        # Define feature categories
        self.numerical_features = []
        self.categorical_features = config.CATEGORICAL_COLS if config.CATEGORICAL_COLS else []
        
        # Drift detection thresholds
        self.drift_threshold = 0.1  # p-value threshold for statistical tests
        self.performance_threshold = 0.05  # Relative performance drop threshold
        
        # Create monitoring directory
        self.monitoring_dir = Path("monitoring_reports")
        self.monitoring_dir.mkdir(exist_ok=True)
        
    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set or update reference data for drift detection."""
        self.reference_data = reference_data
        self._identify_feature_types()
        
    def _identify_feature_types(self):
        """Identify numerical and categorical features from reference data."""
        if self.reference_data is not None:
            # Identify numerical features (excluding target column)
            numerical_cols = self.reference_data.select_dtypes(include=[np.number]).columns
            self.numerical_features = [col for col in numerical_cols if col != 'churn']
            
            # Update categorical features if not set
            if not self.categorical_features:
                categorical_cols = self.reference_data.select_dtypes(include=['object', 'category']).columns
                self.categorical_features = [col for col in categorical_cols if col != 'churn']
    
    def detect_data_drift(self, 
                         current_data: pd.DataFrame, 
                         feature_columns: Optional[List[str]] = None,
                         generate_report: bool = True) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare against reference
            feature_columns: Specific columns to check for drift
            generate_report: Whether to generate detailed Evidently report
            
        Returns:
            Dictionary containing drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Use set_reference_data() first.")
        
        # Prepare data for comparison
        if feature_columns is None:
            feature_columns = self.numerical_features + self.categorical_features
        
        # Ensure both datasets have the same columns
        common_columns = list(set(feature_columns) & set(current_data.columns) & set(self.reference_data.columns))
        
        if not common_columns:
            raise ValueError("No common columns found between reference and current data.")
        
        reference_subset = self.reference_data[common_columns]
        current_subset = current_data[common_columns]
        
        # Statistical drift detection
        drift_results = self._statistical_drift_detection(reference_subset, current_subset)
        
        # Generate Evidently report if requested
        if generate_report:
            report_path = self._generate_evidently_data_drift_report(
                reference_subset, current_subset
            )
            drift_results['report_path'] = report_path
        
        return drift_results
    
    def _statistical_drift_detection(self, 
                                   reference_data: pd.DataFrame, 
                                   current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical drift detection using various tests.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Dictionary containing drift test results
        """
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(reference_data.columns),
            'drifted_features': [],
            'drift_scores': {},
            'drift_detected': False
        }
        
        for column in reference_data.columns:
            if column in current_data.columns:
                # Get data for the column
                ref_data = reference_data[column].dropna()
                curr_data = current_data[column].dropna()
                
                if len(ref_data) == 0 or len(curr_data) == 0:
                    continue
                
                # Choose appropriate test based on data type
                if column in self.numerical_features:
                    drift_score, p_value = self._ks_test(ref_data, curr_data)
                    test_name = 'Kolmogorov-Smirnov'
                else:
                    drift_score, p_value = self._chi_square_test(ref_data, curr_data)
                    test_name = 'Chi-Square'
                
                drift_results['drift_scores'][column] = {
                    'test': test_name,
                    'score': drift_score,
                    'p_value': p_value,
                    'drift_detected': p_value < self.drift_threshold
                }
                
                if p_value < self.drift_threshold:
                    drift_results['drifted_features'].append(column)
        
        drift_results['drift_detected'] = len(drift_results['drifted_features']) > 0
        drift_results['drift_ratio'] = len(drift_results['drifted_features']) / drift_results['total_features']
        
        return drift_results
    
    def _ks_test(self, reference_data: pd.Series, current_data: pd.Series) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for numerical features."""
        try:
            statistic, p_value = stats.ks_2samp(reference_data, current_data)
            return float(statistic), float(p_value)
        except Exception as e:
            self.logger.warning(f"KS test failed: {e}")
            return 0.0, 1.0
    
    def _chi_square_test(self, reference_data: pd.Series, current_data: pd.Series) -> Tuple[float, float]:
        """Perform Chi-square test for categorical features."""
        try:
            # Get value counts for both datasets
            ref_counts = reference_data.value_counts()
            curr_counts = current_data.value_counts()
            
            # Align the categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            # Perform chi-square test
            statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
            return float(statistic), float(p_value)
        except Exception as e:
            self.logger.warning(f"Chi-square test failed: {e}")
            return 0.0, 1.0
    
    def _generate_evidently_data_drift_report(self, 
                                            reference_data: pd.DataFrame, 
                                            current_data: pd.DataFrame) -> str:
        """Generate comprehensive Evidently data drift report."""
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            if 'churn' in reference_data.columns:
                column_mapping.target = 'churn'
            column_mapping.numerical_features = self.numerical_features
            column_mapping.categorical_features = self.categorical_features
            
            # Create and run report
            report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
                DatasetMissingValuesMetric(),
                DatasetSummaryMetric(),
            ])
            
            report.run(reference_data=reference_data, 
                      current_data=current_data, 
                      column_mapping=column_mapping)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.monitoring_dir / f"data_drift_report_{timestamp}.html"
            report.save_html(str(report_path))
            
            self.logger.info(f"Data drift report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Evidently report: {e}")
            return ""
    
    def detect_concept_drift(self, 
                           model_path: str,
                           current_data: pd.DataFrame,
                           current_target: pd.Series,
                           performance_window: int = 100) -> Dict[str, Any]:
        """
        Detect concept drift based on model performance degradation.
        
        Args:
            model_path: Path to the trained model
            current_data: Current feature data
            current_target: True labels for current data
            performance_window: Number of recent predictions to consider
            
        Returns:
            Dictionary containing concept drift results
        """
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Get predictions
            predictions = model.predict(current_data)
            prediction_proba = model.predict_proba(current_data)[:, 1]
            
            # Calculate current performance metrics
            current_metrics = self._calculate_performance_metrics(
                current_target, predictions, prediction_proba
            )
            
            # Load reference performance (if available)
            reference_metrics = self._load_reference_performance()
            
            # Detect drift based on performance comparison
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': current_metrics,
                'reference_metrics': reference_metrics,
                'concept_drift_detected': False,
                'performance_degradation': {}
            }
            
            if reference_metrics:
                # Compare metrics
                for metric_name, current_value in current_metrics.items():
                    if metric_name in reference_metrics:
                        reference_value = reference_metrics[metric_name]
                        
                        # Calculate relative change
                        relative_change = abs(current_value - reference_value) / reference_value
                        
                        drift_results['performance_degradation'][metric_name] = {
                            'current': current_value,
                            'reference': reference_value,
                            'relative_change': relative_change,
                            'drift_detected': relative_change > self.performance_threshold
                        }
                        
                        # Check if drift detected
                        if relative_change > self.performance_threshold:
                            drift_results['concept_drift_detected'] = True
            
            # Save current metrics as reference for future comparisons
            self._save_reference_performance(current_metrics)
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Concept drift detection failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'concept_drift_detected': False
            }
    
    def _calculate_performance_metrics(self, 
                                     y_true: pd.Series, 
                                     y_pred: np.ndarray, 
                                     y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_true, y_proba),
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate some metrics: {e}")
            return {}
    
    def _load_reference_performance(self) -> Optional[Dict[str, float]]:
        """Load reference performance metrics from file."""
        try:
            performance_file = self.monitoring_dir / "reference_performance.json"
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load reference performance: {e}")
        return None
    
    def _save_reference_performance(self, metrics: Dict[str, float]):
        """Save performance metrics as reference for future comparisons."""
        try:
            performance_file = self.monitoring_dir / "reference_performance.json"
            with open(performance_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save reference performance: {e}")
    
    def generate_comprehensive_drift_report(self, 
                                          current_data: pd.DataFrame,
                                          current_target: Optional[pd.Series] = None,
                                          model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report including both data and concept drift.
        
        Args:
            current_data: Current dataset
            current_target: Current target values (for concept drift)
            model_path: Path to model (for concept drift)
            
        Returns:
            Comprehensive drift report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': None,
            'concept_drift': None,
            'overall_drift_detected': False,
            'recommendations': []
        }
        
        # Data drift detection
        try:
            data_drift_results = self.detect_data_drift(current_data)
            report['data_drift'] = data_drift_results
            
            if data_drift_results['drift_detected']:
                report['overall_drift_detected'] = True
                report['recommendations'].append(
                    f"Data drift detected in {len(data_drift_results['drifted_features'])} features. "
                    f"Consider retraining the model or investigating data quality issues."
                )
        except Exception as e:
            self.logger.error(f"Data drift detection failed: {e}")
            report['data_drift'] = {'error': str(e)}
        
        # Concept drift detection (if target and model provided)
        if current_target is not None and model_path is not None:
            try:
                concept_drift_results = self.detect_concept_drift(
                    model_path, current_data, current_target
                )
                report['concept_drift'] = concept_drift_results
                
                if concept_drift_results.get('concept_drift_detected', False):
                    report['overall_drift_detected'] = True
                    report['recommendations'].append(
                        "Concept drift detected based on model performance degradation. "
                        "Model retraining is recommended."
                    )
            except Exception as e:
                self.logger.error(f"Concept drift detection failed: {e}")
                report['concept_drift'] = {'error': str(e)}
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.monitoring_dir / f"comprehensive_drift_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comprehensive drift report saved to {report_path}")
        
        return report


def create_drift_detector(config, reference_data_path: Optional[str] = None) -> DriftDetector:
    """
    Factory function to create and configure a drift detector.
    
    Args:
        config: Configuration object
        reference_data_path: Path to reference data file
        
    Returns:
        Configured DriftDetector instance
    """
    detector = DriftDetector(config)
    
    if reference_data_path and Path(reference_data_path).exists():
        try:
            # Load reference data based on file extension
            if reference_data_path.endswith('.json'):
                reference_data = pd.read_json(reference_data_path)
            elif reference_data_path.endswith('.csv'):
                reference_data = pd.read_csv(reference_data_path)
            else:
                raise ValueError(f"Unsupported file format: {reference_data_path}")
            
            detector.set_reference_data(reference_data)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load reference data: {e}")
    
    return detector
