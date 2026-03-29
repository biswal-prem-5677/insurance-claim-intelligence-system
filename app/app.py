"""Insurance Claim Intelligence System - Web Application
Flask web app for entering new insurance claims and viewing the fraud investigation dashboard.
"""
import sys
import pandas as pd
import subprocess
from flask import Flask, render_template, request, jsonify, send_file, redirect
from pathlib import Path
from datetime import datetime

# Import the decision engine for predictions
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.decision_engine import predict_claim_risk

# Paths - relative to app directory, go up to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
INPUT_CSV = DATA_DIR / "insurance_claims_cleaned.csv"
GRAPH_CSV = DATA_DIR / "insurance_claims_graph_dataset.csv"
DASHBOARD_FILE = PROJECT_ROOT / "fraud_investigation_network.html"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder='../templates', static_folder='../static')


def load_existing_data():
    """Load existing insurance claims data"""
    if INPUT_CSV.exists():
        return pd.read_csv(INPUT_CSV)
    return None


def load_graph_data():
    """Load existing graph dataset"""
    if GRAPH_CSV.exists():
        return pd.read_csv(GRAPH_CSV)
    return None


def get_sample_form_data():
    """Get sample form data structure"""
    return {
        'policy_number': 'POL-001',
        'age': 35,
        'insured_sex': 'MALE',
        'policy_state': 'NC',
        'incident_type': 'Collision',
        'incident_severity': 'Major Damage',
        'total_claim_amount': 25000,
        'auto_make': 'Toyota',
        'auto_model': 'Camry',
        'auto_year': 2018
    }


@app.route('/')
def index():
    """Home page with options"""
    dashboard_exists = DASHBOARD_FILE.exists()
    data_count = 0
    # Don't load the entire CSV just to count rows - it's slow!
    # Just check if file exists and use file size as a quick indication
    if INPUT_CSV.exists():
        try:
            # Quick check: just count lines in the file without loading into memory
            with open(INPUT_CSV, 'r') as f:
                data_count = sum(1 for _ in f) - 1  # -1 for header row
        except Exception:
            data_count = 0  # If reading fails, just show 0
    
    return render_template('index.html', 
                         dashboard_exists=dashboard_exists,
                         data_count=data_count,
                         dashboard_file=str(DASHBOARD_FILE))


@app.route('/add_claim')
def add_claim():
    """Page to add new claim"""
    # Use sensible defaults for dropdown options
    # No need to load CSV file for every form load - it's too slow!
    
    # Common US States for regions
    regions = sorted({
        'NC', 'NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'MI',
        'AZ', 'CO', 'MA', 'WA', 'NJ', 'VA', 'MN', 'MO', 'MD', 'TN'
    })
    
    # Common vehicle makes
    vehicle_types = sorted({
        'Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes-Benz',
        'Nissan', 'Jeep', 'Ram', 'Gmail', 'Volkswagen', 'Mazda', 'Kia',
        'Hyundai', 'Lexus', 'Tesla', 'Audi', 'Acura', 'Infinite', 'Subaru',
        'Dodge', 'Cadillac', 'GMC', 'Buick', 'Chrysler', 'Volvo'
    })
    
    return render_template('add_claim.html', 
                         regions=regions,
                         vehicle_types=vehicle_types,
                         sample=get_sample_form_data())


@app.route('/api/add_claim', methods=['POST'])
def api_add_claim():
    """API endpoint to add new claim to both CSVs"""
    try:
        data = request.json
        
        # Helper function for safe numeric conversion
        def safe_int(val, default=0):
            try:
                if val is None or val == '':
                    return default
                return int(float(val))
            except (TypeError, ValueError):
                return default
        
        def safe_float(val, default=0.0):
            try:
                if val is None or val == '':
                    return default
                return float(val)
            except (TypeError, ValueError):
                return default
        
        # Load templates to get all column structure
        df_cleaned_template = load_existing_data()
        df_graph_template = load_graph_data()
        
        # Create new row for cleaned CSV
        if df_cleaned_template is not None:
            new_row_cleaned = df_cleaned_template.iloc[0].to_dict()
        else:
            new_row_cleaned = {}
        
        # Create new row for graph CSV
        if df_graph_template is not None:
            new_row_graph = df_graph_template.iloc[0].to_dict()
        else:
            new_row_graph = {}
        
        # Generate policy number if not provided
        policy_id = str(data.get('policy_id', '')).strip()
        if not policy_id:
            policy_id = f"POL-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Extract and validate form data with safe conversions
        age = safe_int(data.get('age', 35), 35)
        age = min(max(age, 16), 100)  # Reasonable age range
        
        gender = str(data.get('gender', 'M')).upper().strip()
        gender_char = 'M' if gender in ['M', 'MALE'] else 'F'
        
        region = str(data.get('region', 'OH')).strip()
        vehicle_age = safe_int(data.get('vehicle_age', 5), 5)
        vehicle_age = max(vehicle_age, 0)
        
        vehicle_type = str(data.get('vehicle_type', 'Toyota')).strip()
        claim_amount = safe_float(data.get('claim_amount', 25000), 25000)
        claim_amount = max(claim_amount, 0)
        
        # NEW: Get collision type and injury severity
        collision_type = str(data.get('collision_type', 'Side Collision')).strip()
        injury_severity = str(data.get('injury_severity', 'None')).strip()
        annual_mileage = safe_float(data.get('annual_mileage', 15000), 15000)
        claim_frequency = safe_float(data.get('claim_frequency', 1), 1)
        
        # Create auto_year from vehicle_age
        auto_year = 2024 - vehicle_age
        auto_year = max(auto_year, 1900)
        
        # Update cleaned CSV row with form data
        new_row_cleaned.update({
            'policy_number': policy_id,
            'age': age,
            'insured_sex': gender_char,
            'policy_state': region,
            'incident_state': region,
            'total_claim_amount': claim_amount,
            'auto_make': vehicle_type,
            'auto_year': int(auto_year),
            'collision_type': collision_type,
            'injury_severity_score': 0 if injury_severity.lower() == 'none' else (1 if injury_severity.lower() == 'minor' else 2),
            'bodily_injuries': 0 if injury_severity.lower() == 'none' else (1 if injury_severity.lower() == 'minor' else 2),
            'annual_mileage': annual_mileage,
            'months_as_customer': int(claim_frequency * 12),  # Approximate months
            'fraud_reported': 0
        })
        
        # Update graph CSV row with form data
        new_row_graph.update({
            'policy_number': policy_id,
            'policy_id': policy_id,
            'age': age,
            'insured_sex': gender_char,
            'policy_state': region,
            'incident_state': region,
            'total_claim_amount': claim_amount,
            'auto_make': vehicle_type,
            'auto_year': int(auto_year),
            'collision_type': collision_type,
            'bodily_injuries': 0 if injury_severity.lower() == 'none' else (1 if injury_severity.lower() == 'minor' else 2),
            'fraud_reported': 0
        })
        
        # Save to cleaned CSV
        if INPUT_CSV.exists():
            df_cleaned = pd.read_csv(INPUT_CSV)
            df_cleaned = pd.concat([df_cleaned, pd.DataFrame([new_row_cleaned])], ignore_index=True)
        else:
            df_cleaned = pd.DataFrame([new_row_cleaned])
        
        df_cleaned.to_csv(INPUT_CSV, index=False)
        print(f"✓ Saved claim {policy_id} to {INPUT_CSV.name}")
        
        # Save to graph CSV
        if GRAPH_CSV.exists():
            df_graph = pd.read_csv(GRAPH_CSV)
            df_graph = pd.concat([df_graph, pd.DataFrame([new_row_graph])], ignore_index=True)
        else:
            df_graph = pd.DataFrame([new_row_graph])
        
        df_graph.to_csv(GRAPH_CSV, index=False)
        print(f"✓ Saved claim {policy_id} to {GRAPH_CSV.name}")
        
        return jsonify({
            'success': True,
            'message': f'✅ Claim {policy_id} added successfully to system!',
            'total_claims': len(df_cleaned),
            'policy_id': policy_id
        })
    except Exception as e:
        print(f"❌ Error adding claim: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/regenerate_graph', methods=['POST'])
def regenerate_graph():
    """Regenerate the fraud investigation graph"""
    try:
        print("Regenerating fraud investigation network...")
        result = subprocess.run(
            [sys.executable, 'src/fraud_investigation_system.py'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Graph regenerated successfully!',
                'dashboard_file': str(DASHBOARD_FILE)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Error regenerating graph: {result.stderr}'
            }), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/view_dashboard')
def view_dashboard():
    """Redirect to /view_graph (primary route for fraud investigation graph)"""
    return redirect('/view_graph')


@app.route('/view_graph')
def view_graph():
    """PRIMARY ROUTE: View the fraud investigation graph/dashboard"""
    if DASHBOARD_FILE.exists():
        return send_file(DASHBOARD_FILE, mimetype='text/html')
    return "Graph not found. Please regenerate it first.", 404


@app.route('/api/stats')
def api_stats():
    """Get dataset statistics"""
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)
        return jsonify({
            'total_claims': len(df),
            'avg_claim_amount': float(df['total_claim_amount'].mean()) if 'total_claim_amount' in df.columns else 0,
            'max_claim_amount': float(df['total_claim_amount'].max()) if 'total_claim_amount' in df.columns else 0,
            'states': df['policy_state'].nunique() if 'policy_state' in df.columns else 0,
            'incident_types': df['incident_type'].nunique() if 'incident_type' in df.columns else 0
        })
    return jsonify({'error': 'No data available'}), 404


@app.route('/api/predict_claim', methods=['POST'])
def api_predict_claim():
    """API endpoint to predict fraud risk for a claim"""
    try:
        # Read JSON from frontend
        data = request.get_json()
        data['age'] = float(data.get('age', 0))
        data['vehicle_age'] = float(data.get('vehicle_age', 0))
        data['annual_mileage'] = float(data.get('annual_mileage', 0))
        data['claim_amount'] = float(data.get('claim_amount', 0))
        data['claim_frequency'] = float(data.get('claim_frequency', 0))
        
        # Call the decision engine to get prediction
        result = predict_claim_risk(data)
        
        return jsonify({
            'success': True,
            'fraud_probability': result['fraud_probability'],
            'risk_level': result['risk_level'],
            'reasons': result['reasons']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("Starting Insurance Claim Intelligence System...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
