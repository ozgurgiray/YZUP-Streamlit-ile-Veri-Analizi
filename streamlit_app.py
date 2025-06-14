import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prediction_system import BreastCancerPredictor
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #0d47a1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

# Load model function
@st.cache_resource
def load_model():
    try:
        predictor = BreastCancerPredictor()
        if os.path.exists('breast_cancer_predictor.pkl'):
            predictor.load_model('breast_cancer_predictor.pkl')
            return predictor, True
        else:
            # Train the model if pickle doesn't exist
            predictor.train('data.csv')
            predictor.save_model()
            return predictor, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
        # Clean data
        if 'Unnamed: 32' in df.columns:
            df.drop(['Unnamed: 32'], axis=1, inplace=True)
        if 'id' in df.columns:
            df.drop(['id'], axis=1, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Breast Cancer Diagnosis Assistant</h1>', unsafe_allow_html=True)
    
    # Load model and data
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            st.session_state.predictor, st.session_state.model_loaded = load_model()
    
    df = load_data()
    
    if not st.session_state.model_loaded or df is None:
        st.error("Unable to load model or data. Please check your files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔬 Make Prediction", "📊 Data Explorer", "📈 Model Performance", "📁 Batch Prediction"]
    )
    
    # Import additional page functions
    from streamlit_pages import show_data_explorer, show_model_performance, show_batch_prediction
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "🔬 Make Prediction":
        show_prediction_page()
    elif page == "📊 Data Explorer":
        show_data_explorer(df)
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "📁 Batch Prediction":
        show_batch_prediction()

def show_home_page(df):
    """Home page with overview"""
    st.markdown('<h2 class="sub-header">Welcome to the Breast Cancer Diagnosis Assistant</h2>', unsafe_allow_html=True)
    
    # Overview
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Purpose</h3>
    This streamlit application assists in breast cancer diagnosis using machine learning analysis of cell nucleus features.
    The system achieves <strong>98.25% accuracy</strong> using advanced algorithms trained on the Wisconsin Breast Cancer dataset.
    </div>
    """, unsafe_allow_html=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Samples", len(df))
    
    with col2:
        benign_count = len(df[df['diagnosis'] == 'B'])
        st.metric("✅ Benign Cases", benign_count)
    
    with col3:
        malignant_count = len(df[df['diagnosis'] == 'M'])
        st.metric("⚠️ Malignant Cases", malignant_count)
    
    with col4:
        st.metric("🎯 Model Accuracy", "98.25%")
    
    # Dataset overview
    st.markdown('<h3 class="sub-header">📋 Dataset Overview</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution pie chart
        target_counts = df['diagnosis'].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=['Benign', 'Malignant'],
            title="Diagnosis Distribution",
            color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature categories
        st.markdown("""
        **📏 Feature Categories:**
        - **Mean values**: Average of cell nucleus measurements
        - **Standard Error (SE)**: Variability in measurements  
        - **Worst values**: Largest/most severe measurements
        
        **🔬 Measured Properties:**
        - Radius, Texture, Perimeter, Area
        - Smoothness, Compactness, Concavity
        - Concave Points, Symmetry, Fractal Dimension
        """)
    
    # Model information
    st.markdown('<h3 class="sub-header">🤖 Model Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🏆 Best Model:** Logistic Regression with PCA
        - **Accuracy:** 98.25%
        - **Sensitivity:** 95.24%
        - **Specificity:** 100%
        - **F1-Score:** 97.56%
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Technical Details:**
        - **Algorithm:** Logistic Regression
        - **Feature Reduction:** PCA (7 components)
        - **Preprocessing:** RobustScaler
        - **Cross-Validation:** 97.14% ± 1.49%
        """)
    
    # Quick start guide
    st.markdown('<h3 class="sub-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **🔬 Make Prediction**: Enter cell measurements for individual diagnosis
    2. **📊 Data Explorer**: Explore the dataset with interactive visualizations  
    3. **📈 Model Performance**: View detailed model metrics and comparisons
    4. **📁 Batch Prediction**: Upload CSV files for multiple predictions
    """)

def show_prediction_page():
    """Individual prediction page"""
    st.markdown('<h2 class="sub-header">🔬 Individual Diagnosis Prediction</h2>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Manual Input", "📋 Example Cases", "🎲 Random Sample"]
    )
    
    # Get feature names
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    # Feature input
    features = {}
    
    if input_method == "📝 Manual Input":
        st.markdown("**Enter cell nucleus measurements:**")
        
        # Organize features by category
        categories = {
            "Mean Values": feature_names[:10],
            "Standard Error (SE)": feature_names[10:20], 
            "Worst Values": feature_names[20:]
        }
        
        for category, cat_features in categories.items():
            with st.expander(f"📊 {category}", expanded=True):
                cols = st.columns(2)
                for i, feature in enumerate(cat_features):
                    with cols[i % 2]:
                        # Set reasonable default ranges based on dataset
                        if 'radius' in feature:
                            default_val = 14.0
                            min_val, max_val = 6.0, 30.0
                        elif 'texture' in feature:
                            default_val = 19.0
                            min_val, max_val = 9.0, 40.0
                        elif 'perimeter' in feature:
                            default_val = 92.0
                            min_val, max_val = 40.0, 200.0
                        elif 'area' in feature:
                            default_val = 655.0
                            min_val, max_val = 140.0, 2500.0
                        else:
                            default_val = 0.1
                            min_val, max_val = 0.0, 1.0
                        
                        features[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=0.01,
                            key=f"input_{feature}"
                        )
    
    elif input_method == "📋 Example Cases":
        example_type = st.selectbox(
            "Select example case:",
            ["Typical Benign Case", "Typical Malignant Case", "Borderline Case"]
        )
        
        if example_type == "Typical Benign Case":
            features = {
                'radius_mean': 12.0, 'texture_mean': 18.0, 'perimeter_mean': 80.0,
                'area_mean': 500.0, 'smoothness_mean': 0.09, 'compactness_mean': 0.08,
                'concavity_mean': 0.05, 'concave points_mean': 0.03, 'symmetry_mean': 0.18,
                'fractal_dimension_mean': 0.06, 'radius_se': 0.3, 'texture_se': 1.0,
                'perimeter_se': 2.0, 'area_se': 20.0, 'smoothness_se': 0.005,
                'compactness_se': 0.02, 'concavity_se': 0.02, 'concave points_se': 0.01,
                'symmetry_se': 0.02, 'fractal_dimension_se': 0.003, 'radius_worst': 13.0,
                'texture_worst': 25.0, 'perimeter_worst': 90.0, 'area_worst': 600.0,
                'smoothness_worst': 0.12, 'compactness_worst': 0.15, 'concavity_worst': 0.10,
                'concave points_worst': 0.08, 'symmetry_worst': 0.25, 'fractal_dimension_worst': 0.08
            }
        elif example_type == "Typical Malignant Case":
            features = {
                'radius_mean': 18.0, 'texture_mean': 25.0, 'perimeter_mean': 120.0,
                'area_mean': 1000.0, 'smoothness_mean': 0.12, 'compactness_mean': 0.25,
                'concavity_mean': 0.30, 'concave points_mean': 0.15, 'symmetry_mean': 0.24,
                'fractal_dimension_mean': 0.08, 'radius_se': 1.0, 'texture_se': 2.0,
                'perimeter_se': 8.0, 'area_se': 150.0, 'smoothness_se': 0.008,
                'compactness_se': 0.05, 'concavity_se': 0.05, 'concave points_se': 0.02,
                'symmetry_se': 0.03, 'fractal_dimension_se': 0.006, 'radius_worst': 25.0,
                'texture_worst': 35.0, 'perimeter_worst': 160.0, 'area_worst': 2000.0,
                'smoothness_worst': 0.16, 'compactness_worst': 0.65, 'concavity_worst': 0.70,
                'concave points_worst': 0.25, 'symmetry_worst': 0.45, 'fractal_dimension_worst': 0.12
            }
        else:  # Borderline case
            features = {
                'radius_mean': 15.0, 'texture_mean': 22.0, 'perimeter_mean': 100.0,
                'area_mean': 750.0, 'smoothness_mean': 0.105, 'compactness_mean': 0.15,
                'concavity_mean': 0.15, 'concave points_mean': 0.08, 'symmetry_mean': 0.21,
                'fractal_dimension_mean': 0.07, 'radius_se': 0.6, 'texture_se': 1.5,
                'perimeter_se': 4.0, 'area_se': 70.0, 'smoothness_se': 0.006,
                'compactness_se': 0.03, 'concavity_se': 0.03, 'concave points_se': 0.015,
                'symmetry_se': 0.025, 'fractal_dimension_se': 0.004, 'radius_worst': 18.0,
                'texture_worst': 30.0, 'perimeter_worst': 120.0, 'area_worst': 1200.0,
                'smoothness_worst': 0.14, 'compactness_worst': 0.35, 'concavity_worst': 0.40,
                'concave points_worst': 0.15, 'symmetry_worst': 0.35, 'fractal_dimension_worst': 0.10
            }
    
    else:  # Random sample
        df = load_data()
        if st.button("🎲 Generate Random Sample"):
            sample_idx = np.random.randint(0, len(df))
            sample_row = df.drop('diagnosis', axis=1).iloc[sample_idx]
            for feature in feature_names:
                features[feature] = float(sample_row[feature])
            st.success(f"Generated random sample #{sample_idx}")
        
        # Show input fields for random sample
        if features:
            st.markdown("**Generated values (you can modify them):**")
            cols = st.columns(3)
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    features[feature] = st.number_input(
                        feature.replace('_', ' ').title(),
                        value=features.get(feature, 0.0),
                        step=0.01,
                        key=f"random_{feature}"
                    )
    
    # Make prediction
    if st.button("🔮 Predict Diagnosis", type="primary"):
        if features and len(features) == 30:
            try:
                result = st.session_state.predictor.predict_single(features)
                
                # Display prediction result
                prediction_class = "benign" if result['prediction'] == 'Benign' else "malignant"
                
                st.markdown(f"""
                <div class="prediction-result {prediction_class}">
                    🔬 Prediction: {result['prediction']}
                    <br>
                    🎯 Confidence: {result['confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Probability Breakdown:**")
                    st.metric("Benign Probability", f"{result['probability_benign']:.1%}")
                    st.metric("Malignant Probability", f"{result['probability_malignant']:.1%}")
                
                with col2:
                    # Probability visualization
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Benign', 'Malignant'],
                            y=[result['probability_benign'], result['probability_malignant']],
                            marker_color=['#2E8B57' if result['prediction'] == 'Benign' else '#90EE90',
                                        '#DC143C' if result['prediction'] == 'Malignant' else '#FFB6C1']
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Probabilities",
                        yaxis_title="Probability",
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clinical interpretation
                st.markdown("**🏥 Clinical Interpretation:**")
                if result['prediction'] == 'Benign':
                    if result['confidence'] > 0.9:
                        st.success("High confidence benign prediction. Low likelihood of malignancy.")
                    else:
                        st.warning("Moderate confidence benign prediction. Consider additional testing.")
                else:
                    if result['confidence'] > 0.9:
                        st.error("High confidence malignant prediction. Immediate medical attention recommended.")
                    else:
                        st.warning("Moderate confidence malignant prediction. Further diagnostic workup needed.")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.error("Please fill in all feature values.")

if __name__ == "__main__":
    main()