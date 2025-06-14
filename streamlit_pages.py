import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def show_data_explorer(df):
    """Data exploration page with interactive visualizations"""
    st.markdown('<h2 class="sub-header">📊 Dataset Explorer</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.markdown("### 🔍 Data Preview")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("**Basic Statistics:**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Benign cases: {len(df[df['diagnosis'] == 'B'])}")
        st.write(f"Malignant cases: {len(df[df['diagnosis'] == 'M'])}")
    
    # Feature analysis
    st.markdown("### 📈 Feature Analysis")
    
    # Feature selector
    feature_categories = {
        "Mean Features": [col for col in df.columns if col.endswith('_mean')],
        "SE Features": [col for col in df.columns if col.endswith('_se')],
        "Worst Features": [col for col in df.columns if col.endswith('_worst')]
    }
    
    selected_category = st.selectbox("Select feature category:", list(feature_categories.keys()))
    selected_features = feature_categories[selected_category]
    
    # Visualization type
    viz_type = st.radio(
        "Choose visualization:",
        ["📊 Distribution Plots", "📈 Box Plots", "🔥 Correlation Heatmap", "🎯 Scatter Plot Matrix"]
    )
    
    if viz_type == "📊 Distribution Plots":
        # Distribution plots
        feature = st.selectbox("Select feature for distribution:", selected_features)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Overall Distribution', 'Distribution by Diagnosis']
        )
        
        # Overall distribution
        fig.add_trace(
            go.Histogram(x=df[feature], name='Overall', nbinsx=30),
            row=1, col=1
        )
        
        # Distribution by diagnosis
        for diagnosis in ['B', 'M']:
            subset = df[df['diagnosis'] == diagnosis]
            fig.add_trace(
                go.Histogram(
                    x=subset[feature], 
                    name=f'{"Benign" if diagnosis == "B" else "Malignant"}',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("**📊 Statistical Summary:**")
        summary_stats = df.groupby('diagnosis')[feature].describe().round(3)
        st.dataframe(summary_stats)
    
    elif viz_type == "📈 Box Plots":
        # Box plots
        num_features = min(6, len(selected_features))
        selected_for_box = st.multiselect(
            "Select features for box plots:",
            selected_features,
            default=selected_features[:num_features]
        )
        
        if selected_for_box:
            fig = make_subplots(
                rows=(len(selected_for_box) + 2) // 3,
                cols=3,
                subplot_titles=selected_for_box
            )
            
            for i, feature in enumerate(selected_for_box):
                row = i // 3 + 1
                col = i % 3 + 1
                
                for diagnosis in ['B', 'M']:
                    subset = df[df['diagnosis'] == diagnosis]
                    fig.add_trace(
                        go.Box(
                            y=subset[feature],
                            name=f'{"Benign" if diagnosis == "B" else "Malignant"}',
                            showlegend=(i == 0)
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(height=200 * ((len(selected_for_box) + 2) // 3))
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "🔥 Correlation Heatmap":
        # Correlation heatmap
        # Encode diagnosis for correlation
        df_encoded = df.copy()
        df_encoded['diagnosis'] = df_encoded['diagnosis'].map({'B': 0, 'M': 1})
        
        correlation_features = selected_features + ['diagnosis']
        corr_matrix = df_encoded[correlation_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            title=f"Correlation Heatmap - {selected_category}",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Highest correlations with diagnosis
        st.markdown("**🎯 Features Most Correlated with Diagnosis:**")
        diagnosis_corr = corr_matrix['diagnosis'].abs().sort_values(ascending=False)[1:6]
        for feature, corr_val in diagnosis_corr.items():
            st.write(f"• {feature}: {corr_val:.3f}")
    
    elif viz_type == "🎯 Scatter Plot Matrix":
        # Scatter plot matrix
        num_features = min(4, len(selected_features))
        selected_for_scatter = st.multiselect(
            "Select features for scatter matrix (max 4):",
            selected_features,
            default=selected_features[:num_features]
        )
        
        if selected_for_scatter:
            scatter_df = df[selected_for_scatter + ['diagnosis']].copy()
            scatter_df['diagnosis'] = scatter_df['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})
            
            fig = px.scatter_matrix(
                scatter_df,
                dimensions=selected_for_scatter,
                color='diagnosis',
                color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'},
                title="Feature Relationships"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🏆 Feature Importance Analysis")
    
    if st.button("🔍 Analyze Feature Importance"):
        with st.spinner("Calculating feature importance..."):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data
            X = df.drop('diagnosis', axis=1)
            y = LabelEncoder().fit_transform(df['diagnosis'])
            
            # Train Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features",
                labels={'importance': 'Feature Importance', 'feature': 'Features'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features
            st.markdown("**🥇 Top 10 Features:**")
            st.dataframe(importance_df.head(10), use_container_width=True)

def show_model_performance():
    """Model performance dashboard"""
    st.markdown('<h2 class="sub-header">📈 Model Performance Dashboard</h2>', unsafe_allow_html=True)
    
    # Load results if available
    try:
        results_df = pd.read_csv('ml_models_results.csv')
        
        # Performance overview
        st.markdown("### 🎯 Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_accuracy = results_df['Accuracy'].max()
        best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        avg_accuracy = results_df['Accuracy'].mean()
        total_models = len(results_df)
        
        with col1:
            st.metric("🏆 Best Accuracy", f"{best_accuracy:.1%}")
        with col2:
            st.metric("🤖 Best Model", best_model.split('_')[0])
        with col3:
            st.metric("📊 Average Accuracy", f"{avg_accuracy:.1%}")
        with col4:
            st.metric("🔢 Models Tested", total_models)
        
        # Model comparison
        st.markdown("### 📊 Model Comparison")
        
        # Performance by model type
        model_types = results_df['Model'].str.split('_').str[0]
        performance_by_type = results_df.groupby(model_types).agg({
            'Accuracy': 'mean',
            'Sensitivity': 'mean',
            'Specificity': 'mean',
            'F1_Score': 'mean'
        }).round(4)
        
        fig = px.bar(
            performance_by_type.reset_index(),
            x='Model',
            y=['Accuracy', 'Sensitivity', 'Specificity', 'F1_Score'],
            title="Average Performance by Model Type",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature set comparison
        st.markdown("### 🔧 Feature Set Impact")
        
        feature_performance = results_df.groupby('Feature_Set').agg({
            'Accuracy': ['mean', 'std'],
            'Train_Time': 'mean'
        }).round(4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy by feature set
            fig = px.box(
                results_df,
                x='Feature_Set',
                y='Accuracy',
                title="Accuracy Distribution by Feature Set"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training time by feature set
            fig = px.bar(
                results_df.groupby('Feature_Set')['Train_Time'].mean().reset_index(),
                x='Feature_Set',
                y='Train_Time',
                title="Average Training Time by Feature Set"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        st.markdown("### 🏆 Best Model Analysis")
        
        best_model_idx = results_df['Accuracy'].idxmax()
        best_model_data = results_df.iloc[best_model_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Performance Metrics:**")
            metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score']
            for metric in metrics:
                if metric in best_model_data:
                    st.metric(metric, f"{best_model_data[metric]:.1%}")
        
        with col2:
            # Performance radar chart
            metrics_values = [best_model_data[m] for m in metrics if m in best_model_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=metrics_values,
                theta=metrics,
                fill='toself',
                name='Best Model'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="Best Model Performance Radar"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        # Sort by accuracy
        display_df = results_df.sort_values('Accuracy', ascending=False)
        
        # Format percentage columns
        percentage_cols = ['Accuracy', 'Sensitivity', 'Specificity', 'F1_Score', 'CV_Mean']
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download results
        if st.button("📥 Download Results"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
    
    except FileNotFoundError:
        st.warning("Model results file not found. Please run the model training first.")
        
        if st.button("🚀 Run Model Training"):
            with st.spinner("Training models... This may take a few minutes."):
                import subprocess
                subprocess.run(["python", "ml_models_training.py"])
                st.success("Model training completed! Please refresh the page.")

def show_batch_prediction():
    """Batch prediction page"""
    st.markdown('<h2 class="sub-header">📁 Batch Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file with breast cancer cell measurements to get predictions for multiple samples.
    The CSV should contain the same 30 features used in individual predictions.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with the 30 feature columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Validate columns
            required_features = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
            ]
            
            missing_features = set(required_features) - set(batch_df.columns)
            extra_features = set(batch_df.columns) - set(required_features) - {'id', 'diagnosis'}
            
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                return
            
            if extra_features:
                st.warning(f"Extra features found (will be ignored): {extra_features}")
            
            # Prepare data for prediction
            features_df = batch_df[required_features]
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("📊 Samples to Predict", len(features_df))
            with col2:
                st.metric("✅ Valid Features", len(required_features))
            
            # Make predictions
            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        results = st.session_state.predictor.predict_batch(features_df)
                        
                        # Combine with original data
                        if 'id' in batch_df.columns:
                            results['id'] = batch_df['id']
                            results = results[['id'] + [col for col in results.columns if col != 'id']]
                        
                        # Add original diagnosis if available
                        if 'diagnosis' in batch_df.columns:
                            results['actual_diagnosis'] = batch_df['diagnosis']
                            # Calculate accuracy
                            if len(results) > 0:
                                # Map predictions for comparison
                                pred_mapped = results['prediction'].map({'Benign': 'B', 'Malignant': 'M'})
                                accuracy = (pred_mapped == results['actual_diagnosis']).mean()
                                st.success(f"Prediction accuracy on uploaded data: {accuracy:.1%}")
                        
                        st.markdown("### 🎯 Prediction Results")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            benign_count = len(results[results['prediction'] == 'Benign'])
                            st.metric("✅ Predicted Benign", benign_count)
                        with col2:
                            malignant_count = len(results[results['prediction'] == 'Malignant'])
                            st.metric("⚠️ Predicted Malignant", malignant_count)
                        with col3:
                            avg_confidence = results['confidence'].mean()
                            st.metric("🎯 Average Confidence", f"{avg_confidence:.1%}")
                        
                        # Results visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Prediction distribution
                            pred_counts = results['prediction'].value_counts()
                            fig = px.pie(
                                values=pred_counts.values,
                                names=pred_counts.index,
                                title="Prediction Distribution",
                                color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            fig = px.histogram(
                                results,
                                x='confidence',
                                color='prediction',
                                title="Confidence Score Distribution",
                                color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results table
                        st.markdown("### 📋 Detailed Results")
                        
                        # Format display
                        display_results = results.copy()
                        display_results['confidence'] = display_results['confidence'].apply(lambda x: f"{x:.1%}")
                        display_results['probability_benign'] = display_results['probability_benign'].apply(lambda x: f"{x:.1%}")
                        display_results['probability_malignant'] = display_results['probability_malignant'].apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(display_results, use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    else:
        # Show sample format
        st.markdown("### 📝 Sample CSV Format")
        
        st.markdown("""
        Your CSV file should contain columns with the following names:
        """)
        
        sample_data = {
            'radius_mean': [14.0, 12.5, 18.2],
            'texture_mean': [19.5, 16.8, 24.1],
            'perimeter_mean': [92.3, 85.1, 115.6],
            '...': ['...', '...', '...'],
            'fractal_dimension_worst': [0.08, 0.07, 0.12]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("""
        **📋 Requirements:**
        - Must contain all 30 feature columns
        - Optional: 'id' column for sample identification
        - Optional: 'diagnosis' column (B/M) for accuracy calculation
        - No missing values in feature columns
        """)
        
        # Download sample template
        if st.button("📥 Download Sample Template"):
            # Create sample template
            feature_names = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
            ]
            
            template_df = pd.DataFrame(columns=['id'] + feature_names + ['diagnosis'])
            csv = template_df.to_csv(index=False)
            
            st.download_button(
                label="📁 Download Template",
                data=csv,
                file_name="batch_prediction_template.csv",
                mime="text/csv"
            )