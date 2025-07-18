"""
Monitoring Script for Customer Churn Prediction Pipeline

This script provides scheduled monitoring capabilities including:
- Automated drift detection
- Alert generation
- Performance tracking
- Integration with existing pipeline
"""

import sys
import os
import logging
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from monitoring.drift_detection import create_drift_detector
from utils.logging_config import setup_logging


class MonitoringOrchestrator:
    """
    Orchestrates the monitoring pipeline including drift detection and alerting.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the monitoring orchestrator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize drift detector
        self.drift_detector = create_drift_detector(
            config, 
            reference_data_path=config.TRAINING_DATA_PATH
        )
        
        # Monitoring configuration
        self.monitoring_config = {
            'drift_check_interval': 24,  # hours
            'performance_check_interval': 6,  # hours
            'alert_threshold': 0.1,  # drift threshold for alerts
            'email_alerts': getattr(config, 'EMAIL_ALERTS_ENABLED', False),
            'slack_alerts': getattr(config, 'SLACK_ALERTS_ENABLED', False),
        }
        
        # Alert configuration
        self.alert_config = {
            'email': {
                'smtp_server': getattr(config, 'SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': getattr(config, 'SMTP_PORT', 587),
                'sender_email': getattr(config, 'SENDER_EMAIL', ''),
                'sender_password': getattr(config, 'SENDER_PASSWORD', ''),
                'recipient_emails': getattr(config, 'RECIPIENT_EMAILS', []),
            },
            'slack': {
                'webhook_url': getattr(config, 'SLACK_WEBHOOK_URL', ''),
                'channel': getattr(config, 'SLACK_CHANNEL', '#ml-monitoring'),
            }
        }
        
        # Create monitoring logs directory
        self.monitoring_logs_dir = Path("logs/monitoring")
        self.monitoring_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logger for monitoring
        self._setup_monitoring_logger()
    
    def _setup_monitoring_logger(self):
        """Set up dedicated logger for monitoring activities."""
        handler = logging.FileHandler(
            self.monitoring_logs_dir / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        )
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def check_data_drift(self, current_data_path: str) -> Dict[str, Any]:
        """
        Check for data drift using current data.
        
        Args:
            current_data_path: Path to current data file
            
        Returns:
            Drift detection results
        """
        try:
            # Load current data
            if current_data_path.endswith('.json'):
                current_data = pd.read_json(current_data_path)
            elif current_data_path.endswith('.csv'):
                current_data = pd.read_csv(current_data_path)
            else:
                raise ValueError(f"Unsupported file format: {current_data_path}")
            
            # Detect drift
            drift_results = self.drift_detector.detect_data_drift(current_data)
            
            # Log results
            self.logger.info(f"Data drift check completed. Drift detected: {drift_results['drift_detected']}")
            
            if drift_results['drift_detected']:
                self.logger.warning(
                    f"Data drift detected in {len(drift_results['drifted_features'])} features: "
                    f"{drift_results['drifted_features']}"
                )
                
                # Send alerts if configured
                if self.monitoring_config['email_alerts']:
                    self._send_drift_alert(drift_results, 'data_drift')
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Data drift check failed: {e}")
            return {'error': str(e), 'drift_detected': False}
    
    def check_concept_drift(self, 
                           current_data_path: str, 
                           current_target_path: str,
                           model_path: str) -> Dict[str, Any]:
        """
        Check for concept drift using current data and model.
        
        Args:
            current_data_path: Path to current data file
            current_target_path: Path to current target file
            model_path: Path to trained model
            
        Returns:
            Concept drift detection results
        """
        try:
            # Load current data and target
            if current_data_path.endswith('.json'):
                current_data = pd.read_json(current_data_path)
            elif current_data_path.endswith('.csv'):
                current_data = pd.read_csv(current_data_path)
            else:
                raise ValueError(f"Unsupported file format: {current_data_path}")
            
            if current_target_path.endswith('.json'):
                current_target = pd.read_json(current_target_path)
            elif current_target_path.endswith('.csv'):
                current_target = pd.read_csv(current_target_path)
            else:
                raise ValueError(f"Unsupported file format: {current_target_path}")
            
            # Ensure target is a Series
            if isinstance(current_target, pd.DataFrame):
                current_target = current_target.iloc[:, 0]
            
            # Detect concept drift
            concept_drift_results = self.drift_detector.detect_concept_drift(
                model_path, current_data, current_target
            )
            
            # Log results
            self.logger.info(
                f"Concept drift check completed. Drift detected: "
                f"{concept_drift_results.get('concept_drift_detected', False)}"
            )
            
            if concept_drift_results.get('concept_drift_detected', False):
                self.logger.warning("Concept drift detected - model performance degradation")
                
                # Send alerts if configured
                if self.monitoring_config['email_alerts']:
                    self._send_drift_alert(concept_drift_results, 'concept_drift')
            
            return concept_drift_results
            
        except Exception as e:
            self.logger.error(f"Concept drift check failed: {e}")
            return {'error': str(e), 'concept_drift_detected': False}
    
    def run_comprehensive_monitoring(self, 
                                   current_data_path: str,
                                   current_target_path: Optional[str] = None,
                                   model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive monitoring including both data and concept drift detection.
        
        Args:
            current_data_path: Path to current data file
            current_target_path: Path to current target file (optional)
            model_path: Path to trained model (optional)
            
        Returns:
            Comprehensive monitoring results
        """
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': None,
            'concept_drift': None,
            'overall_status': 'healthy',
            'actions_required': []
        }
        
        try:
            # Load current data
            if current_data_path.endswith('.json'):
                current_data = pd.read_json(current_data_path)
            elif current_data_path.endswith('.csv'):
                current_data = pd.read_csv(current_data_path)
            else:
                raise ValueError(f"Unsupported file format: {current_data_path}")
            
            # Load current target if provided
            current_target = None
            if current_target_path:
                if current_target_path.endswith('.json'):
                    current_target = pd.read_json(current_target_path)
                elif current_target_path.endswith('.csv'):
                    current_target = pd.read_csv(current_target_path)
                
                if isinstance(current_target, pd.DataFrame):
                    current_target = current_target.iloc[:, 0]
            
            # Generate comprehensive drift report
            comprehensive_report = self.drift_detector.generate_comprehensive_drift_report(
                current_data, current_target, model_path
            )
            
            monitoring_results.update(comprehensive_report)
            
            # Determine overall status and actions
            if comprehensive_report.get('overall_drift_detected', False):
                monitoring_results['overall_status'] = 'drift_detected'
                monitoring_results['actions_required'] = comprehensive_report.get('recommendations', [])
                
                # Send comprehensive alert
                if self.monitoring_config['email_alerts']:
                    self._send_comprehensive_alert(comprehensive_report)
            
            # Log comprehensive results
            self.logger.info(f"Comprehensive monitoring completed. Status: {monitoring_results['overall_status']}")
            
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive monitoring failed: {e}")
            monitoring_results['error'] = str(e)
            monitoring_results['overall_status'] = 'error'
            return monitoring_results
    
    def _send_drift_alert(self, drift_results: Dict[str, Any], drift_type: str):
        """Send drift alert via email."""
        try:
            if not self.alert_config['email']['sender_email'] or not self.alert_config['email']['recipient_emails']:
                self.logger.warning("Email alert configuration incomplete")
                return
            
            # Prepare email content
            subject = f"[ML Monitoring] {drift_type.replace('_', ' ').title()} Detected"
            
            if drift_type == 'data_drift':
                body = f"""
                Data Drift Alert
                
                Timestamp: {drift_results.get('timestamp', 'Unknown')}
                Drift Detected: {drift_results.get('drift_detected', False)}
                Drifted Features: {drift_results.get('drifted_features', [])}
                Total Features: {drift_results.get('total_features', 0)}
                Drift Ratio: {drift_results.get('drift_ratio', 0):.2%}
                
                Please investigate the data quality and consider model retraining.
                
                Report Path: {drift_results.get('report_path', 'N/A')}
                """
            else:  # concept_drift
                body = f"""
                Concept Drift Alert
                
                Timestamp: {drift_results.get('timestamp', 'Unknown')}
                Concept Drift Detected: {drift_results.get('concept_drift_detected', False)}
                
                Performance Degradation:
                """
                
                for metric, details in drift_results.get('performance_degradation', {}).items():
                    body += f"""
                {metric.title()}:
                  Current: {details.get('current', 'N/A'):.4f}
                  Reference: {details.get('reference', 'N/A'):.4f}
                  Change: {details.get('relative_change', 0):.2%}
                  Drift: {details.get('drift_detected', False)}
                """
                
                body += "\nModel retraining is recommended."
            
            # Send email
            self._send_email(subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send drift alert: {e}")
    
    def _send_comprehensive_alert(self, comprehensive_report: Dict[str, Any]):
        """Send comprehensive monitoring alert."""
        try:
            if not self.alert_config['email']['sender_email'] or not self.alert_config['email']['recipient_emails']:
                self.logger.warning("Email alert configuration incomplete")
                return
            
            subject = "[ML Monitoring] Comprehensive Drift Report"
            
            body = f"""
            Comprehensive Drift Monitoring Report
            
            Timestamp: {comprehensive_report.get('timestamp', 'Unknown')}
            Overall Drift Detected: {comprehensive_report.get('overall_drift_detected', False)}
            
            Data Drift Summary:
            {json.dumps(comprehensive_report.get('data_drift', {}), indent=2)}
            
            Concept Drift Summary:
            {json.dumps(comprehensive_report.get('concept_drift', {}), indent=2)}
            
            Recommendations:
            """
            
            for rec in comprehensive_report.get('recommendations', []):
                body += f"- {rec}\n"
            
            self._send_email(subject, body)
            
        except Exception as e:
            self.logger.error(f"Failed to send comprehensive alert: {e}")
    
    def _send_email(self, subject: str, body: str):
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.alert_config['email']['sender_email']
            msg['To'] = ', '.join(self.alert_config['email']['recipient_emails'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                self.alert_config['email']['smtp_server'],
                self.alert_config['email']['smtp_port']
            )
            server.starttls()
            server.login(
                self.alert_config['email']['sender_email'],
                self.alert_config['email']['sender_password']
            )
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent successfully: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
    
    def schedule_monitoring(self, 
                          current_data_path: str,
                          current_target_path: Optional[str] = None,
                          model_path: Optional[str] = None):
        """
        Schedule periodic monitoring tasks.
        
        Args:
            current_data_path: Path to current data file
            current_target_path: Path to current target file (optional)
            model_path: Path to trained model (optional)
        """
        # Schedule data drift checks
        schedule.every(self.monitoring_config['drift_check_interval']).hours.do(
            self.check_data_drift, current_data_path
        )
        
        # Schedule concept drift checks (if model and target provided)
        if model_path and current_target_path:
            schedule.every(self.monitoring_config['performance_check_interval']).hours.do(
                self.check_concept_drift, current_data_path, current_target_path, model_path
            )
        
        # Schedule comprehensive monitoring
        schedule.every(24).hours.do(  # Daily comprehensive check
            self.run_comprehensive_monitoring, current_data_path, current_target_path, model_path
        )
        
        self.logger.info("Monitoring tasks scheduled successfully")
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main function for running monitoring."""
    parser = argparse.ArgumentParser(description='ML Pipeline Monitoring')
    parser.add_argument('--mode', choices=['data_drift', 'concept_drift', 'comprehensive', 'schedule'], 
                       default='comprehensive', help='Monitoring mode')
    parser.add_argument('--current-data', required=True, help='Path to current data file')
    parser.add_argument('--current-target', help='Path to current target file')
    parser.add_argument('--model-path', help='Path to trained model')
    parser.add_argument('--config', default='config.py', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = Config()
    
    # Initialize monitoring orchestrator
    orchestrator = MonitoringOrchestrator(config)
    
    # Run monitoring based on mode
    if args.mode == 'data_drift':
        results = orchestrator.check_data_drift(args.current_data)
        print(json.dumps(results, indent=2))
        
    elif args.mode == 'concept_drift':
        if not args.current_target or not args.model_path:
            print("Error: --current-target and --model-path required for concept drift detection")
            sys.exit(1)
        
        results = orchestrator.check_concept_drift(
            args.current_data, args.current_target, args.model_path
        )
        print(json.dumps(results, indent=2))
        
    elif args.mode == 'comprehensive':
        results = orchestrator.run_comprehensive_monitoring(
            args.current_data, args.current_target, args.model_path
        )
        print(json.dumps(results, indent=2))
        
    elif args.mode == 'schedule':
        print("Starting scheduled monitoring...")
        orchestrator.schedule_monitoring(
            args.current_data, args.current_target, args.model_path
        )


if __name__ == "__main__":
    main()
