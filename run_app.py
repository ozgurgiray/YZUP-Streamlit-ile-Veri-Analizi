#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    required_packages = [
        'streamlit', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    return True

def check_model_files():
    required_files = ['data.csv', 'prediction_system.py', 'streamlit_app.py', 'streamlit_pages.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    return True

def train_model_if_needed():
    if not os.path.exists('breast_cancer_predictor.pkl'):
        print("🚀 Training model for first-time setup...")
        try:
            subprocess.run([sys.executable, 'prediction_system.py'], check=True)
            print("✅ Model training completed!")
        except subprocess.CalledProcessError:
            print("❌ Model training failed!")
            return False
    return True

def launch_streamlit():
    print("🚀 Launching Breast Cancer Diagnosis Assistant...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   • Use the sidebar to navigate between pages")
    print("   • Try the example cases in the prediction page")
    print("   • Explore the data visualizations")
    print("   • Check model performance metrics")
    print("\n⏹️  Press Ctrl+C to stop the application\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.headless', 'false',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped!")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

def main():
    print("=" * 60)
    print("🏥 BREAST CANCER DIAGNOSIS ASSISTANT")
    print("🤖 AI-Powered Medical Screening Tool")
    print("=" * 60)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All packages available")
    
    # Check files
    print("📁 Checking project files...")
    if not check_model_files():
        sys.exit(1)
    print("✅ All files present")
    
    # Train model if needed
    print("🤖 Checking model status...")
    if not train_model_if_needed():
        sys.exit(1)
    print("✅ Model ready")
    
    # Launch application
    launch_streamlit()

if __name__ == "__main__":
    main()