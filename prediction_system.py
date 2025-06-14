import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class BreastCancerPredictor:
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_data(self, df):
        # Clean data
        if 'Unnamed: 32' in df.columns:
            df.drop(['Unnamed: 32'], axis=1, inplace=True)
        if 'id' in df.columns:
            df.drop(['id'], axis=1, inplace=True)
        
        return df
    
    def train(self, csv_file_path='data.csv'):
        print("Training Breast Cancer Predictor...")
        print("-" * 40)
        
        # Load and prepare data
        df = pd.read_csv(csv_file_path)
        df = self.prepare_data(df)
        
        # Encode target
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M:1, B:0
        
        # Split features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA (90% variance - 7 components)
        self.pca = PCA(n_components=7, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Train the best model (Logistic Regression)
        self.model = LogisticRegression(C=10, max_iter=1000, penalty='l2', random_state=42)
        self.model.fit(X_train_pca, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_pca)
        y_pred_proba = self.model.predict_proba(X_test_pca)
        
        # Print training results
        print(f"Training completed!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Original features: {len(self.feature_names)}")
        print(f"PCA components: {self.pca.n_components_}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {(y_pred == y_test).mean():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        self.is_trained = True
        print("\nModel trained successfully!")
        
        return {
            'accuracy': (y_pred == y_test).mean(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def predict_single(self, features):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Convert to DataFrame if it's a dict
        if isinstance(features, dict):
            # Ensure all required features are present
            if set(features.keys()) != set(self.feature_names):
                missing = set(self.feature_names) - set(features.keys())
                extra = set(features.keys()) - set(self.feature_names)
                if missing:
                    raise ValueError(f"Missing features: {missing}")
                if extra:
                    raise ValueError(f"Extra features: {extra}")
            
            # Create DataFrame with correct order
            sample = pd.DataFrame([features])[self.feature_names]
        else:
            # Assume it's a list/array in correct order
            if len(features) != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
            sample = pd.DataFrame([features], columns=self.feature_names)
        
        # Apply same preprocessing
        sample_scaled = self.scaler.transform(sample)
        sample_pca = self.pca.transform(sample_scaled)
        
        # Make prediction
        prediction = self.model.predict(sample_pca)[0]
        probability = self.model.predict_proba(sample_pca)[0]
        
        result = {
            'prediction': 'Malignant' if prediction == 1 else 'Benign',
            'prediction_code': int(prediction),
            'probability_benign': float(probability[0]),
            'probability_malignant': float(probability[1]),
            'confidence': float(max(probability))
        }
        
        return result
    
    def predict_batch(self, features_df):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Ensure correct feature order
        features_df = features_df[self.feature_names]
        
        # Apply preprocessing
        features_scaled = self.scaler.transform(features_df)
        features_pca = self.pca.transform(features_scaled)
        
        # Make predictions
        predictions = self.model.predict(features_pca)
        probabilities = self.model.predict_proba(features_pca)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions],
            'prediction_code': predictions,
            'probability_benign': probabilities[:, 0],
            'probability_malignant': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1)
        })
        
        return results
    
    def save_model(self, filepath='breast_cancer_predictor.pkl'):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='breast_cancer_predictor.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get PCA components
        components = self.pca.components_
        
        # Calculate feature importance by summing absolute values across components
        feature_importance = np.sum(np.abs(components), axis=0)
        
        # Normalize
        feature_importance = feature_importance / feature_importance.sum()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Example usage and testing
if __name__ == "__main__":
    print("=== BREAST CANCER PREDICTION SYSTEM ===\n")
    
    # Initialize predictor
    predictor = BreastCancerPredictor()
    
    # Train the model
    training_results = predictor.train('data.csv')
    
    # Save the model
    predictor.save_model()
    
    print("\n" + "="*50)
    print("TESTING PREDICTION SYSTEM")
    print("="*50)
    
    # Load test data for demonstration
    df = pd.read_csv('data.csv')
    df = predictor.prepare_data(df)
    
    # Test single prediction with first sample
    print("\n1. Single Prediction Test:")
    print("-" * 30)
    
    # Get a sample from the dataset
    sample_features = df.drop('diagnosis', axis=1).iloc[0].to_dict()
    actual_diagnosis = df.iloc[0]['diagnosis']
    
    result = predictor.predict_single(sample_features)
    
    print(f"Actual diagnosis: {actual_diagnosis}")
    print(f"Predicted: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probability Benign: {result['probability_benign']:.4f}")
    print(f"Probability Malignant: {result['probability_malignant']:.4f}")
    
    # Test batch prediction
    print("\n2. Batch Prediction Test:")
    print("-" * 30)
    
    # Test on first 10 samples
    test_features = df.drop('diagnosis', axis=1).head(10)
    batch_results = predictor.predict_batch(test_features)
    
    print("First 10 predictions:")
    comparison = pd.DataFrame({
        'Actual': df.head(10)['diagnosis'].values,
        'Predicted': batch_results['prediction'].values,
        'Confidence': batch_results['confidence'].values
    })
    print(comparison.to_string(index=False))
    
    # Feature importance
    print("\n3. Feature Importance:")
    print("-" * 30)
    importance = predictor.get_feature_importance()
    print("Top 10 most important features:")
    print(importance.head(10).to_string(index=False))
    
    # Example with custom features (using mean values)
    print("\n4. Custom Sample Prediction:")
    print("-" * 30)
    
    # Create a sample with typical benign characteristics
    benign_sample = {
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
    
    benign_result = predictor.predict_single(benign_sample)
    print(f"Benign-like sample prediction: {benign_result['prediction']}")
    print(f"Confidence: {benign_result['confidence']:.4f}")
    
    print("\n=== PREDICTION SYSTEM READY FOR USE ===")
    print("Use predictor.predict_single() or predictor.predict_batch() for new predictions")