import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                           roc_curve, precision_recall_curve, auc, roc_auc_score)
import time
import warnings
warnings.filterwarnings('ignore')

# Reproduce the preprocessing from previous script
def prepare_data():
    """Prepare and preprocess the breast cancer data"""
    df = pd.read_csv('data.csv')
    
    # Clean data
    if 'Unnamed: 32' in df.columns:
        df.drop(['Unnamed: 32'], axis=1, inplace=True)
    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)
    
    # Encode target
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M:1, B:0
    
    # Split data
    random_state = 42
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare different feature sets
    # PCA 90% variance (7 components)
    pca_90 = PCA(n_components=7, random_state=random_state)
    X_train_pca90 = pca_90.fit_transform(X_train_scaled)
    X_test_pca90 = pca_90.transform(X_test_scaled)
    
    # PCA 95% variance (10 components)
    pca_95 = PCA(n_components=10, random_state=random_state)
    X_train_pca95 = pca_95.fit_transform(X_train_scaled)
    X_test_pca95 = pca_95.transform(X_test_scaled)
    
    # RFE selected features
    rfecv = RFECV(cv=StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True),
                  estimator=DecisionTreeClassifier(random_state=random_state), 
                  scoring='accuracy')
    rfecv.fit(X_train_scaled, y_train)
    X_train_rfe = X_train_scaled[:, rfecv.get_support()]
    X_test_rfe = X_test_scaled[:, rfecv.get_support()]
    
    feature_sets = {
        'Original': (X_train_scaled, X_test_scaled),
        'PCA_90%': (X_train_pca90, X_test_pca90),
        'PCA_95%': (X_train_pca95, X_test_pca95),
        'RFE': (X_train_rfe, X_test_rfe)
    }
    
    return feature_sets, y_train, y_test, X.columns

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)  # Recall/True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    precision = tp / (tp + fp)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_time': train_time,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def hyperparameter_tuning(model_class, param_grid, X_train, y_train, model_name):
    """Perform hyperparameter tuning using GridSearchCV"""
    print(f"\nTuning hyperparameters for {model_name}...")
    
    grid_search = GridSearchCV(
        estimator=model_class(),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

print("=== MACHINE LEARNING MODELS TRAINING AND EVALUATION ===\n")

# Prepare data
feature_sets, y_train, y_test, feature_names = prepare_data()
print("Data preparation completed!")

# Define models and their hyperparameters
models_config = {
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'random_state': [42],
            'max_iter': [1000]
        }
    },
    'Random Forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'random_state': [42]
        }
    },
    'SVM': {
        'class': SVC,
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'probability': [True],
            'random_state': [42]
        }
    },
    'K-Nearest Neighbors': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    'Decision Tree': {
        'class': DecisionTreeClassifier,
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }
    }
}

# Store all results
all_results = []

# Train and evaluate models on different feature sets
for feature_set_name, (X_train_fs, X_test_fs) in feature_sets.items():
    print(f"\n{'='*60}")
    print(f"FEATURE SET: {feature_set_name} ({X_train_fs.shape[1]} features)")
    print(f"{'='*60}")
    
    for model_name, config in models_config.items():
        print(f"\n--- {model_name} ---")
        
        # Hyperparameter tuning
        best_model, best_params = hyperparameter_tuning(
            config['class'], config['params'], X_train_fs, y_train, model_name
        )
        
        # Evaluate model
        results = evaluate_model(
            best_model, X_train_fs, X_test_fs, y_train, y_test, 
            f"{model_name}_{feature_set_name}"
        )
        
        results['feature_set'] = feature_set_name
        results['best_params'] = best_params
        all_results.append(results)
        
        # Print results
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Sensitivity: {results['sensitivity']:.4f}")
        print(f"Specificity: {results['specificity']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        if results['roc_auc']:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        print(f"Training Time: {results['train_time']:.4f}s")

# Create comprehensive results comparison
print(f"\n{'='*80}")
print("COMPREHENSIVE RESULTS COMPARISON")
print(f"{'='*80}")

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'Feature_Set': r['feature_set'],
        'Accuracy': r['accuracy'],
        'Sensitivity': r['sensitivity'],
        'Specificity': r['specificity'],
        'Precision': r['precision'],
        'F1_Score': r['f1_score'],
        'ROC_AUC': r['roc_auc'],
        'CV_Mean': r['cv_mean'],
        'CV_Std': r['cv_std'],
        'Train_Time': r['train_time']
    }
    for r in all_results
])

# Display top 10 results by accuracy
print("\nTop 10 Models by Test Accuracy:")
print("-" * 50)
top_models = results_df.nlargest(10, 'Accuracy')
print(top_models[['Model', 'Feature_Set', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_Score']].to_string(index=False))

# Find best model overall
best_model_idx = results_df['Accuracy'].idxmax()
best_result = all_results[best_model_idx]

print(f"\n{'='*50}")
print("BEST PERFORMING MODEL")
print(f"{'='*50}")
print(f"Model: {best_result['model_name']}")
print(f"Feature Set: {best_result['feature_set']}")
print(f"Best Parameters: {best_result['best_params']}")
print(f"Test Accuracy: {best_result['accuracy']:.4f}")
print(f"Sensitivity: {best_result['sensitivity']:.4f}")
print(f"Specificity: {best_result['specificity']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"F1-Score: {best_result['f1_score']:.4f}")
if best_result['roc_auc']:
    print(f"ROC AUC: {best_result['roc_auc']:.4f}")

# Confusion Matrix for best model
print(f"\nConfusion Matrix:")
print(best_result['confusion_matrix'])

# Visualizations
plt.figure(figsize=(20, 15))

# 1. Model performance comparison
plt.subplot(3, 3, 1)
pivot_accuracy = results_df.pivot(index='Model', columns='Feature_Set', values='Accuracy')
sns.heatmap(pivot_accuracy, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Accuracy Comparison Across Models and Feature Sets')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 2. Best models by feature set
plt.subplot(3, 3, 2)
best_by_feature = results_df.loc[results_df.groupby('Feature_Set')['Accuracy'].idxmax()]
plt.bar(best_by_feature['Feature_Set'], best_by_feature['Accuracy'])
plt.title('Best Accuracy by Feature Set')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')

# 3. Training time comparison
plt.subplot(3, 3, 3)
plt.scatter(results_df['Train_Time'], results_df['Accuracy'], 
           c=pd.Categorical(results_df['Feature_Set']).codes, alpha=0.7)
plt.xlabel('Training Time (seconds)')
plt.ylabel('Accuracy')
plt.title('Training Time vs Accuracy')

# 4. Sensitivity vs Specificity
plt.subplot(3, 3, 4)
plt.scatter(results_df['Sensitivity'], results_df['Specificity'], 
           c=pd.Categorical(results_df['Model']).codes, alpha=0.7)
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')
plt.title('Sensitivity vs Specificity')

# 5. F1-Score distribution
plt.subplot(3, 3, 5)
results_df.boxplot(column='F1_Score', by='Feature_Set', ax=plt.gca())
plt.title('F1-Score Distribution by Feature Set')
plt.suptitle('')

# 6. Cross-validation scores
plt.subplot(3, 3, 6)
plt.errorbar(range(len(results_df)), results_df['CV_Mean'], 
             yerr=results_df['CV_Std'], fmt='o', alpha=0.7)
plt.xlabel('Model Index')
plt.ylabel('CV Score')
plt.title('Cross-Validation Scores with Standard Deviation')

# 7. Best model confusion matrix
plt.subplot(3, 3, 7)
sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_result["model_name"]}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 8. ROC Curve for best model (if available)
if best_result['y_pred_proba'] is not None:
    plt.subplot(3, 3, 8)
    fpr, tpr, _ = roc_curve(y_test, best_result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {best_result["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend()

# 9. Feature set performance summary
plt.subplot(3, 3, 9)
feature_performance = results_df.groupby('Feature_Set')['Accuracy'].mean()
plt.bar(feature_performance.index, feature_performance.values)
plt.title('Average Accuracy by Feature Set')
plt.xticks(rotation=45)
plt.ylabel('Average Accuracy')

plt.tight_layout()
plt.savefig('ml_models_comprehensive_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results to CSV
results_df.to_csv('ml_models_results.csv', index=False)
print(f"\n=== ANALYSIS COMPLETE ===")
print("Results saved to 'ml_models_results.csv'")
print("Visualizations saved to 'ml_models_comprehensive_results.png'")

print(f"\n=== SUMMARY ===")
print(f"• Trained {len(models_config)} different algorithms")
print(f"• Tested on {len(feature_sets)} different feature sets")
print(f"• Total {len(all_results)} model configurations evaluated")
print(f"• Best model: {best_result['model_name']} with {best_result['feature_set']} features")
print(f"• Best accuracy: {best_result['accuracy']:.4f}")