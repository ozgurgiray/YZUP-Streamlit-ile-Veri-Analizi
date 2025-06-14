# 🏥 YZUP - Breast Cancer Classification with Streamlit Project - Özgür Giray

## 📖 Project Overview

This project implements a breast cancer classification system using machine learning. The system analyzes cell nucleus features to predict whether a tumor is benign or malignant with **98.25% accuracy**.

## 🎯 What This Project Does

- **Analyzes** breast cancer cell measurements
- **Predicts** benign vs malignant tumors
- **Provides** interactive web interface for diagnosis
- **Achieves** 98.25% accuracy using advanced ML algorithms
- **Offers** batch processing for multiple samples

## 📁 Project Files & What They Do

### 🔬 **Core Analysis Files**
| File | Purpose | What it does |
|------|---------|--------------|
| `data.csv` | Dataset | Wisconsin Breast Cancer dataset (569 samples, 30 features) |
| `breast_cancer_analysis.py` | Data Analysis | Comprehensive data exploration and visualization |
| `ml_models_training.py` | Model Training | Trains and compares 5 ML algorithms on 4 feature sets |
| `prediction_system.py` | ML Model | Production-ready predictor class with best model |

### 🌐 **Web Application Files**
| File | Purpose | What it does |
|------|---------|--------------|
| `streamlit_app.py` | Main Web App | Home page and prediction interface |
| `streamlit_pages.py` | Additional Pages | Data explorer, performance dashboard, batch prediction |
| `run_app.py` | App Launcher | Easy script to start the web application |

### 📋 **Documentation & Setup**
| File | Purpose | What it does |
|------|---------|--------------|
| `README.md` | This file | Simple overview and instructions |
| `requirements.txt` | Dependencies | List of required Python packages |

### 📊 **Generated Files** (Auto-created)
| File | Purpose | What it contains |
|------|---------|------------------|
| `breast_cancer_predictor.pkl` | Trained Model | Saved ML model for predictions |
| `ml_models_results.csv` | Results Data | Detailed performance comparison |
| `*.png` files | Visualizations | Analysis plots and charts |

## 🚀 How to Run This Project

### **Option 1: Run Web Application (Recommended)**
```bash
# Quick start - launches web interface
python run_app.py
```
Then open your browser to `http://localhost:8501`

### **Option 2: Step-by-Step Analysis**
```bash
# 1. Data analysis and preprocessing
python breast_cancer_analysis.py

# 2. Train and compare models
python ml_models_training.py

# 3. Test prediction system
python prediction_system.py

# 4. Launch web app
streamlit run streamlit_app.py
```

### **Option 3: Direct Streamlit**
```bash
streamlit run streamlit_app.py
```

## 🛠️ Setup Requirements

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn`, `plotly` - Visualizations
- `numpy` - Numerical computing

## 🎯 Project Results Summary

### 🏆 Best Model Performance
- **Algorithm**: Logistic Regression with PCA
- **Accuracy**: 98.25%
- **Sensitivity**: 95.24% (detects 95% of malignant cases)
- **Specificity**: 100% (no false alarms)
- **Features Used**: 7 PCA components (reduced from 30 original features)

### 📊 Key Findings
- **Dataset**: 569 samples (62.7% benign, 37.3% malignant)
- **Best Feature Reduction**: PCA with 90% variance (7 components)
- **Top Features**: area_worst, concave points_worst, radius_worst
- **Models Tested**: 5 algorithms × 4 feature sets = 20 configurations

## 🌐 Application Features

### 📱 **5 Interactive Pages**

1. **🏠 Home** - Project overview and statistics
2. **🔬 Prediction** - Individual tumor diagnosis
3. **📊 Data Explorer** - Interactive dataset analysis
4. **📈 Performance** - Model comparison dashboard  
5. **📁 Batch Processing** - Multiple sample predictions

### 🎨 **User Interface**
- Real-time predictions with confidence scores
- Interactive charts and visualizations
- Export/download capabilities

## 📋 Usage Examples

### Individual Prediction
1. Open web app: `python run_app.py`
2. Go to "🔬 Make Prediction"
3. Enter cell measurements or use examples
4. Get instant diagnosis with confidence score

### Batch Processing
1. Prepare CSV with 30 feature columns
2. Upload in "📁 Batch Prediction" page
3. Download results with predictions

### Data Analysis
1. Run `python breast_cancer_analysis.py`
2. Explore generated visualizations
3. Review statistical summaries

### **Data Source**
- Wisconsin Breast Cancer (Diagnostic) Dataset# YZUP-Streamlit-ile-Veri-Analizi
