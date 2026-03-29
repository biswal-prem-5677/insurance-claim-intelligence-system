# Insurance Claim Intelligence System - Complete Guide

## System Overview

A comprehensive fraud investigation system with AI-powered graph analysis and interactive visualization dashboard.

### Key Features
- **Graph-based Fraud Detection**: 2,608 nodes representing policies, vehicles, locations, and ZIP codes
- **Web Interface**: Add new claims and regenerate graphs
- **Interactive Dashboard**: Visual fraud investigation network
- **Advanced Filters**: Search by entity, apply graph metrics
- **User-Friendly Tooltips**: Click/hover on nodes for formatted information

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Fraud Detection Model

**First-time setup**: Train XGBoost model (selected as optimal via comprehensive model comparison):

```bash
python src/train_model.py
```

**What this does:**
- Loads the feature-engineered dataset
- Trains XGBoost with optimal hyperparameters
- Saves artifacts: `models/fraud_model.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl`
- Takes ~10 seconds

**Note**: Only needs to run once initially or monthly when new training data arrives.

### 3. Start the Web Application

```bash
python app.py
```

Open your browser to: http://localhost:5000

---

## 🎯 Fraud Detection Model — XGBoost Selected

**Status**: ✅ Model selection completed and validated with aggressive regularization  
**Report**: See [MODEL_SELECTION_REPORT.md](MODEL_SELECTION_REPORT.md) for detailed analysis

### Model Performance Metrics (Final with Aggressive Regularization)
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **F1 Score** | **0.7304** | Excellent balance of precision & recall |
| **Recall** | **85.71%** | Catches vast majority of fraud attempts |
| **Precision** | **63.64%** | Reasonable false positive rate |
| **ROC-AUC** | 0.8348 | Excellent fraud/legitimate discrimination |
| **Train/Test Gap** | **3.25%** | ✅ Well-generalised (excellent) |

### Previous Performance (Before Regularization)
- F1 Score: 0.6061 (39.39% overfitting gap)
- **Improvement**: +20.4% F1 score increase after aggressive regularization
- **Overfitting Reduction**: 91.8% gap reduction (39.39% → 3.25%)

### Why XGBoost?
- ✅ **Best F1 Score** among 4 evaluated models (+20.8% vs Logistic Regression)
- ✅ **Optimal Recall** — catches 85.7% of fraud without overwhelming investigators
- ✅ **Excellent Generalization** — 3.25% train/test gap (well-generalised model)
- ✅ **Production Ready** — <1 sec training, ~0.01ms inference per claim
- ✅ **Explainable** — provides feature importance for investigator insights
- ✅ **Industry Standard** — battle-tested in financial fraud detection
- ✅ **Aggressive Regularization** — prevents overfitting with strong L1/L2 penalties

### Regularization Optimization
The model uses aggressive regularization parameters to prevent overfitting:
- `max_depth=3` (very shallow trees)
- `learning_rate=0.03` (ultra-conservative learning)
- `reg_alpha=2.0, reg_lambda=2.5` (strong weight penalties)
- `min_child_weight=10` (requires more samples per leaf)
- Result: Model generalizes excellently to unseen fraud cases

### Models Evaluated & Rejected
| Model | Final F1 | Why Not Selected |
|-------|----------|-----------------|
| Logistic Regression | 0.5821 | 20.8% lower F1 than XGBoost |
| Random Forest | 0.5057 | 44.1% lower F1, misses too much fraud |
| Support Vector Machine | 0.5785 | 26.3% lower F1, slower inference |

---

## Using the System

### Web Application (app.py)

When you run `app.py`, a Flask web interface opens with three main sections:

#### **Home Page**
- Shows total claims in dataset
- Statistics about the graph
- Buttons to:
  - ➕ **Add New Claim** - Enter new insurance data
  - 📊 **View Dashboard** - Open fraud investigation dashboard
  - 🔄 **Regenerate Graph** - Update with new data

#### **Add New Claim**
- Fill in policy details:
  - Policy ID (auto-generated if empty)
  - Age, Gender, Region
  - Vehicle information (age, type, mileage)
  - Claim amount ($) and frequency
- Data is automatically appended to the dataset
- Saved to: `data/processed/insurance_claims_cleaned.csv`

#### **Fraud Investigation Dashboard**
- Interactive PyVis network graph
- Left panel with filters and controls
- Real-time node filtering
- Physics simulation controls

---

## Dashboard Features

### Node Types (Color & Shape)

| Entity Type | Shape | Color | Represents |
|-----------|-------|-------|-----------|
| Policy Holder | Circle | Blue | People who filed claims |
| Vehicle | Triangle | Orange | Cars/Vehicles involved |
| Location | Square | Green | Geographic incident locations |
| ZIP Code | Diamond | Purple | Postal code regions |
| **Fraud Case** | Any | **Red border** | Connected to confirmed fraud |

### Filter Panel (Left Side)

#### **Checkbox Filters**
1. **Show Fraud Cases** - Only show nodes connected to fraud_reported = 1
2. **Shared Vehicles** - Vehicles used by multiple policyholders (fraud rings)
3. **Suspicious Nodes** - Nodes with high degree_centrality (network hubs)
   - Threshold: default 0.05 (5% importance)
4. **Frequent Accident Locations** - Locations with incident_count > N
5. **High Claim Connections** - Relationships with claims > $threshold

#### **Text Search Filters**
- **Search by Vehicle Name**: e.g., "SUV", "Sedan", "Two wheeler"
- **Search by Location**: e.g., "Downtown", "Airport", "Highway"
- **Search by ZIP Code**: e.g., "10001", "90210"

#### **Node Limit Slider**
- Show top 10-500 most important nodes (by degree centrality)
- Helps focus investigation on key actors

#### **Physics Controls**
- ❄️ **Freeze Layout** - Lock node positions
- ▶️ **Enable Physics** - Allow nodes to move/rearrange

#### **Action Buttons**
- **Apply Filters** - Apply all active filters
- **Reset** - Clear all filters, show full graph

---

## Understanding Node Information (Hover/Click)

When you hover over a node, a formatted tooltip appears with:

### Policy Holder Nodes 🧑
- Policy ID
- Name/Label
- Connections (relationships count)
- Network Importance (0-100%)
- Fraud Risk
- Shared Entities score

### Vehicle Nodes 🚗
- Vehicle ID
- Type (e.g., SUV, Sedan)
- Shared by N policies
- Risk Factor
- Frequency Score

### Location Nodes 📍
- Location ID
- Name
- Claims reported from location
- Incident Rate
- Frequency Score

### ZIP Code Nodes 📮
- ZIP Code
- Area Name
- Claims Activity count
- Concentration (%)
- Risk Level (0-10)

---

## Workflow Example

### Scenario: Investigate Suspected Fraud Ring

1. **Open Dashboard** (view_dashboard button)
2. **Enable Filters**:
   - ✅ Check "Shared Vehicles" - find cars used by multiple people
   - ✅ Check "Suspicious Nodes" - find central network hubs
   - Set Suspicious threshold to 0.10
3. **Apply Filters** - see filtered network
4. **Hover over Nodes** - read formatted info
5. **Examine Connections** - look for patterns:
   - Red edges = fraud claims
   - Thick edges = high-value claims
6. **Use Node Limit** - reduce to top 50 nodes for clarity

---

## Filter Logic Explanation

### What Gets Reduced When Filters Are Applied

| Filter | Node Reduction | Use Case |
|--------|---|---|
| **Show Fraud** | Only fraud-connected nodes remain | Focus on confirmed cases |
| **Shared Vehicles** | Only multi-use vehicles + their links | Find organized fraud rings |
| **Suspicious Nodes** | Only highly connected nodes | Identify fraud network hubs |
| **Frequent Locations** | Only high-incident locations | Find fraud hotspots |
| **High Claim Edges** | Only expensive relationships | Focus on high-value fraud |
| **Text Searches** | Only matching entity names | Find specific vehicles/locations/zips |

---

## System Architecture

```
Insurance_Claim_Intelligence_System/
├── app.py                              # Flask web app for data entry
├── src/
│   ├── fraud_investigation_system.py   # Core graph generator
│   ├── decision_engine.py              # Fraud detection logic
│   ├── explainability.py               # Feature attribution
│   └── train_model.py                  # Model training
├── data/
│   ├── processed/
│   │   ├── insurance_claims_cleaned.csv
│   │   ├── insurance_claims_feature_engineered.csv
│   │   └── insurance_claims_graph_dataset.csv
│   └── raw/                            # Original data
├── templates/
│   ├── index.html                      # Home page
│   ├── add_claim.html                  # Data entry form
├── fraud_investigation_network.html    # Generated dashboard
└── requirements.txt
```

---

## Technical Details

### Graph Composition
- **Nodes**: 2,608
  - ~1,000 Policy Holders
  - ~564 Vehicles
  - ~49 Locations
  - ~995 ZIP Codes

- **Edges**: 3,000
  - Policy ↔ Vehicle (ownership)
  - Policy ↔ Location (incident)
  - Vehicle ↔ Location (involvement)
  - ZIP ↔ Claims (geographic)

### Fraud Detection Metrics
- **Degree Centrality**: Network importance (0-1)
- **Betweenness Centrality**: Bridge importance in network
- **Shared Entity Score**: Reuse frequency (vehicles/locations)
- **Fraud Flag**: Labels confirmed fraud cases (1 = fraud, 0 = normal)

### Data Pipeline
1. User adds claim in web app
2. Data saved to CSV
3. User clicks "Regenerate Graph"
4. Python runs `src/fraud_investigation_system.py`
5. Graph created with NetworkX
6. PyVis generates interactive visualization
7. Dashboard updates automatically

---

## Troubleshooting

### Dashboard Not Loading
- Regenerate graph: Click "🔄 Regenerate Graph"
- Check console (F12) for errors
- Ensure `fraud_investigation_network.html` exists

### Filters Not Working
- Click "Apply Filters" button
- Check browser console (F12) for JavaScript errors
- Reset filters if stuck

### New Claims Not Appearing
- Click "🔄 Regenerate Graph" after adding claims
- Regeneration takes 30-60 seconds

### Text Search Not Finding Results
- Ensure exact spelling/partial matches
- Search is case-insensitive
- Try shorter search terms

---

## API Endpoints (Flask)

```
GET  /                          # Home page
GET  /add_claim                 # Data entry form
POST /api/add_claim             # Add new claim (JSON)
POST /regenerate_graph          # Regenerate graph
GET  /view_dashboard            # Open dashboard
GET  /api/stats                 # Dataset statistics
```

---

## Support & Documentation

- **Graph Analysis**: See `LAYOUT_TECHNICAL_REFERENCE.md`
- **Model Details**: See notebooks in `notebooks/` folder
- **Live Dashboard**: Must run `python src/fraud_investigation_system.py`

---

**System Created**: 2024  
**Version**: 1.0  
**Status**: Production Ready  
