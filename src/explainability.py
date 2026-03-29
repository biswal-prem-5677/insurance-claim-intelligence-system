"""
Insurance Claim Intelligence System — Model Explainability
==========================================================

Lightweight explainability module for the fraud detection system.
Provides simple feature importance analysis and explanations.

This module focuses on simplicity and performance, avoiding heavy
dependencies like SHAP to keep the system lightweight.

Author: Insurance Claim Intelligence System Team
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "src":
        return script_dir.parent
    return script_dir


PROJECT_ROOT = _get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# Module-level cache for loaded model
_model = None
_feature_columns = None


def _load_model() -> None:
    """Load the trained XGBoost model and feature columns."""
    global _model, _feature_columns
    
    if _model is not None:
        return
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train_model.py first."
        )
    
    _model = joblib.load(MODEL_PATH)
    _feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


def get_feature_importance(top_n: int = 10) -> pd.DataFrame:
    """
    Get feature importance from the trained XGBoost model.
    
    Args:
        top_n (int): Number of top features to return (default: 10)
    
    Returns:
        pd.DataFrame: DataFrame with feature names and importance scores,
                      sorted by importance (descending)
    """
    _load_model()
    
    # Get feature importance from XGBoost model
    importance_scores = _model.feature_importances_
    
    # Create DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({
        'feature': _feature_columns,
        'importance': importance_scores
    })
    
    # Sort by importance (descending) and take top N
    feature_importance_df = feature_importance_df.sort_values(
        'importance', ascending=False
    ).head(top_n)
    
    # Reset index for clean display
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    feature_importance_df.index = range(1, len(feature_importance_df) + 1)
    
    return feature_importance_df


def explain_fraud_risk(input_data: Dict[str, Any], top_features: int = 5) -> Dict[str, Any]:
    """
    Provide a simple explanation for a fraud prediction.
    
    Args:
        input_data (Dict[str, Any]): Input features for the claim
        top_features (int): Number of top contributing features to highlight
    
    Returns:
        Dict[str, Any]: Explanation including key factors and risk interpretation
    """
    _load_model()
    
    # Get overall feature importance
    feature_importance = get_feature_importance(top_features * 2)  # Get more to filter
    
    # Identify which important features are present in the input
    explanation_factors = []
    
    for _, row in feature_importance.iterrows():
        feature_name = row['feature']
        importance = row['importance']
        
        if feature_name in input_data:
            value = input_data[feature_name]
            
            # Create interpretation based on feature name and value
            interpretation = _interpret_feature(feature_name, value, importance)
            if interpretation:
                explanation_factors.append({
                    'feature': feature_name,
                    'value': value,
                    'importance': importance,
                    'interpretation': interpretation
                })
    
    # Sort by importance and take top features
    explanation_factors.sort(key=lambda x: x['importance'], reverse=True)
    explanation_factors = explanation_factors[:top_features]
    
    # Generate overall risk interpretation
    risk_interpretation = _generate_risk_interpretation(input_data, explanation_factors)
    
    return {
        'key_factors': explanation_factors,
        'risk_interpretation': risk_interpretation,
        'model_confidence': 'high' if len(explanation_factors) >= 3 else 'medium'
    }


def _interpret_feature(feature_name: str, value: float, importance: float) -> str:
    """
    Create a human-readable interpretation for a feature value.
    
    Args:
        feature_name (str): Name of the feature
        value (float): Feature value
        importance (float): Feature importance score
    
    Returns:
        str: Human-readable interpretation
    """
    
    # High-impact financial features
    if 'claim_amount' in feature_name.lower() or 'total_claim' in feature_name.lower():
        if value > 0.5:  # High normalized claim amount
            return "High claim amount increases fraud risk"
        elif value > 0.2:
            return "Above-average claim amount"
        else:
            return "Standard claim amount"
    
    elif 'premium' in feature_name.lower():
        if value < -0.5:  # Low normalized premium
            return "Low premium relative to claim increases risk"
        else:
            return "Premium level within normal range"
    
    elif 'ratio' in feature_name.lower():
        if 'claim_to_premium' in feature_name.lower():
            if value > 2.0:
                return "Very high claim-to-premium ratio is suspicious"
            elif value > 1.0:
                return "High claim-to-premium ratio"
            else:
                return "Normal claim-to-premium ratio"
    
    # Severity and incident features
    elif 'severity' in feature_name.lower():
        if value > 2.0:
            return "High incident severity increases fraud risk"
        elif value > 1.0:
            return "Moderate incident severity"
        else:
            return "Low incident severity"
    
    elif 'vehicle' in feature_name.lower():
        if 'number_of_vehicles' in feature_name.lower():
            if value > 2:
                return "Multiple vehicles involved increases complexity"
            else:
                return "Single vehicle incident"
    
    # Temporal features
    elif 'hour' in feature_name.lower():
        if value > 0.8 or value < 0.2:  # Unusual hours
            return "Unusual incident time increases risk"
        else:
            return "Normal incident time"
    
    # Binary indicators
    elif 'indicator' in feature_name.lower() or 'flag' in feature_name.lower():
        if value > 0.5:
            return "Risk flag detected"
        else:
            return "No risk indicators"
    
    # Generic interpretation based on value
    else:
        if abs(value) > 1.0:
            return f"Unusual {feature_name} value detected"
        elif abs(value) > 0.5:
            return f"{feature_name} outside normal range"
        else:
            return f"{feature_name} within expected range"


def _generate_risk_interpretation(input_data: Dict[str, Any], 
                                explanation_factors: List[Dict]) -> str:
    """
    Generate an overall risk interpretation based on key factors.
    
    Args:
        input_data (Dict[str, Any]): Input features
        explanation_factors (List[Dict]): List of key contributing factors
    
    Returns:
        str: Overall risk interpretation
    """
    
    # Count high-risk indicators
    high_risk_count = 0
    medium_risk_count = 0
    
    risk_keywords = {
        'high': ['high', 'suspicious', 'unusual', 'above-average', 'multiple'],
        'medium': ['moderate', 'elevated', 'increases']
    }
    
    for factor in explanation_factors:
        interpretation = factor['interpretation'].lower()
        if any(keyword in interpretation for keyword in risk_keywords['high']):
            high_risk_count += 1
        elif any(keyword in interpretation for keyword in risk_keywords['medium']):
            medium_risk_count += 1
    
    # Generate interpretation based on risk factors
    if high_risk_count >= 2:
        return ("Multiple high-risk factors detected. The claim shows several indicators "
                "commonly associated with fraudulent activity, including unusual financial "
                "patterns and suspicious incident characteristics.")
    
    elif high_risk_count >= 1 or medium_risk_count >= 3:
        return ("Several risk factors present. While not definitively fraudulent, "
                "the claim exhibits characteristics that warrant additional review "
                "and verification of supporting documentation.")
    
    elif medium_risk_count >= 1:
        return ("Some elevated risk factors noted. The claim has minor anomalies "
                "that may deserve closer attention but appears largely consistent "
                "with legitimate claim patterns.")
    
    else:
        return ("Low risk profile. The claim characteristics fall within normal "
                "ranges and show minimal indicators associated with fraudulent activity.")


def get_model_summary() -> Dict[str, Any]:
    """
    Get a summary of the trained model for documentation purposes.
    
    Returns:
        Dict[str, Any]: Model summary including type, features, and performance notes
    """
    _load_model()
    
    return {
        'model_type': type(_model).__name__,
        'total_features': len(_feature_columns),
        'feature_names': _feature_columns[:10],  # First 10 features
        'model_file': str(MODEL_PATH),
        'explainability_method': 'XGBoost Feature Importance',
        'last_updated': 'See model file timestamp',
        'notes': 'XGBoost model trained with class imbalance handling using scale_pos_weight'
    }


# Example usage when run as script
if __name__ == "__main__":
    print("🔍 Insurance Claim Intelligence System - Explainability Module")
    print("=" * 60)
    
    try:
        # Test feature importance
        print("\n📊 Top 10 Most Important Features:")
        importance_df = get_feature_importance(10)
        print(importance_df.to_string())
        
        # Test explanation with sample data
        sample_input = {
            "total_claim_amount": 0.7,
            "claim_to_premium_ratio": 2.5,
            "incident_severity": 2.0,
            "number_of_vehicles_involved": 3,
            "high_claim_indicator": 1
        }
        
        print("\n🎯 Sample Fraud Risk Explanation:")
        explanation = explain_fraud_risk(sample_input)
        print(f"Risk Interpretation: {explanation['risk_interpretation']}")
        print(f"Model Confidence: {explanation['model_confidence']}")
        
        print("\n📋 Key Factors:")
        for i, factor in enumerate(explanation['key_factors'], 1):
            print(f"  {i}. {factor['interpretation']}")
        
        # Model summary
        print("\n🤖 Model Summary:")
        summary = get_model_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model is trained by running src/train_model.py")