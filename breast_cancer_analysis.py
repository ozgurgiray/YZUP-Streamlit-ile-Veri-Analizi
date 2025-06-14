import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

print("=== BREAST CANCER CLASSIFICATION ANALYSIS ===\n")

# Load and Clean Data
print("1. LOADING AND CLEANING DATA")
print("-" * 40)
df = pd.read_csv('data.csv')
print(f"Original dataset shape: {df.shape}")
print(f"Dataset columns: {list(df.columns)}")

# Check for missing values
missing_values = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing_values[missing_values > 0])

# Drop unnecessary columns
if 'Unnamed: 32' in df.columns:
    df.drop(['Unnamed: 32'], axis=1, inplace=True)
if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)
    
print(f"Cleaned dataset shape: {df.shape}")
print(f"Target variable distribution:\n{df['diagnosis'].value_counts()}")

# Exploratory Data Analysis
print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Target distribution
print(f"\nTarget Distribution:")
target_counts = df['diagnosis'].value_counts()
print(f"Benign (B): {target_counts['B']} ({target_counts['B']/len(df)*100:.1f}%)")
print(f"Malignant (M): {target_counts['M']} ({target_counts['M']/len(df)*100:.1f}%)")

# Create comprehensive visualizations
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# Target distribution
plt.subplot(3, 4, 1)
df['diagnosis'].value_counts().plot(kind='bar', color=['lightblue', 'salmon'])
plt.title('Diagnosis Distribution')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Outlier detection with boxplots
mean_features = [col for col in df.columns if col.endswith('_mean')][:6]
for i, feature in enumerate(mean_features):
    plt.subplot(3, 4, i+2)
    df.boxplot(column=feature, by='diagnosis', ax=plt.gca())
    plt.title(f'{feature} by Diagnosis')
    plt.suptitle('')

# Feature distributions
plt.subplot(3, 4, 8)
plt.hist(df[df['diagnosis']=='B']['radius_mean'], alpha=0.7, label='Benign', bins=20)
plt.hist(df[df['diagnosis']=='M']['radius_mean'], alpha=0.7, label='Malignant', bins=20)
plt.title('Radius Mean Distribution')
plt.legend()

plt.subplot(3, 4, 9)
plt.hist(df[df['diagnosis']=='B']['texture_mean'], alpha=0.7, label='Benign', bins=20)
plt.hist(df[df['diagnosis']=='M']['texture_mean'], alpha=0.7, label='Malignant', bins=20)
plt.title('Texture Mean Distribution')
plt.legend()

# Correlation heatmap
plt.subplot(3, 4, 10)
df_encoded = df.copy()
le = LabelEncoder()
df_encoded['diagnosis'] = le.fit_transform(df_encoded['diagnosis'])
correlation_matrix = df_encoded[mean_features + ['diagnosis']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix (Mean Features)')

# Pairplot for selected features
plt.subplot(3, 4, 11)
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
for feature in selected_features:
    plt.scatter(df[df['diagnosis']=='B'][feature], df[df['diagnosis']=='B']['smoothness_mean'], 
               alpha=0.6, label='Benign' if feature == selected_features[0] else "", color='blue')
    plt.scatter(df[df['diagnosis']=='M'][feature], df[df['diagnosis']=='M']['smoothness_mean'], 
               alpha=0.6, label='Malignant' if feature == selected_features[0] else "", color='red')
plt.xlabel('Various Mean Features')
plt.ylabel('Smoothness Mean')
plt.title('Feature Relationships')
plt.legend()

plt.tight_layout()
plt.savefig('comprehensive_eda_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Data Preprocessing and Feature Engineering
print("\n3. DATA PREPROCESSING AND FEATURE ENGINEERING")
print("-" * 50)

# Label encoding
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # M:1, B:0
print(f"Encoded target distribution:\n{df['diagnosis'].value_counts()}")

# Data split
random_state = 42
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")

# Feature scaling using RobustScaler (better for outliers)
print("\nApplying RobustScaler for feature scaling...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction Analysis

# PCA Analysis
print("\nPrincipal Component Analysis (PCA)")
pca = PCA()
pca.fit(X_train_scaled)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(exp_var_cumul)+1), exp_var_cumul, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance')
plt.legend()
plt.grid(True)

# Find components for 90% and 95% variance
n_components_90 = np.argmax(exp_var_cumul >= 0.90) + 1
n_components_95 = np.argmax(exp_var_cumul >= 0.95) + 1
print(f"Components needed for 90% variance: {n_components_90}")
print(f"Components needed for 95% variance: {n_components_95}")

# Feature Importance using Random Forest
print("\nFeature Importance Analysis")
rf_temp = RandomForestClassifier(n_estimators=100, random_state=random_state)
rf_temp.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

plt.subplot(1, 2, 2)
plt.barh(range(10), feature_importance.head(10)['importance'])
plt.yticks(range(10), feature_importance.head(10)['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('dimensionality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Recursive Feature Elimination
print("\nRecursive Feature Elimination (RFE)")
rfecv = RFECV(cv=StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True),
              estimator=DecisionTreeClassifier(random_state=random_state), 
              scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)

print(f"Optimal number of features (RFE): {rfecv.n_features_}")
print("Selected features:")
selected_features = X.columns[rfecv.get_support()]
print(selected_features.tolist())

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
         rfecv.cv_results_['mean_test_score'], 'bo-')
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross Validation Score (Accuracy)')
plt.title('RFE: Number of Features vs CV Score')
plt.grid(True)
plt.savefig('rfe_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Prepare different feature sets for model training
print("\n4. PREPARING FEATURE SETS FOR MODEL TRAINING")
print("-" * 50)

# Original features
X_train_original = X_train_scaled
X_test_original = X_test_scaled

# PCA features (90% variance)
pca_90 = PCA(n_components=n_components_90, random_state=random_state)
X_train_pca90 = pca_90.fit_transform(X_train_scaled)
X_test_pca90 = pca_90.transform(X_test_scaled)

# PCA features (95% variance)
pca_95 = PCA(n_components=n_components_95, random_state=random_state)
X_train_pca95 = pca_95.fit_transform(X_train_scaled)
X_test_pca95 = pca_95.transform(X_test_scaled)

# RFE selected features
X_train_rfe = X_train_scaled[:, rfecv.get_support()]
X_test_rfe = X_test_scaled[:, rfecv.get_support()]

feature_sets = {
    'Original': (X_train_original, X_test_original),
    'PCA_90%': (X_train_pca90, X_test_pca90),
    'PCA_95%': (X_train_pca95, X_test_pca95),
    'RFE': (X_train_rfe, X_test_rfe)
}

print("Feature sets prepared:")
for name, (X_tr, X_te) in feature_sets.items():
    print(f"- {name}: {X_tr.shape[1]} features")

print("\n=== DATA ANALYSIS AND PREPROCESSING COMPLETE ===")
print("Ready for model training and evaluation!")