"""
Setup Script for Indian Road Traffic Accident Prediction System
This script helps set up the environment and initial configuration
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required Python packages"""
    print("Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python packages")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "logs",
        "models",
        "documentation"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Directories created successfully")

def setup_environment():
    """Setup environment file"""
    env_template = Path(".env.template")
    env_file = Path(".env")

    if env_template.exists() and not env_file.exists():
        with open(env_template, "r") as f:
            content = f.read()

        with open(env_file, "w") as f:
            f.write(content)

        print("✅ Environment file created (.env)")
        print("⚠️  Please edit .env file with your actual API keys")
    else:
        print("✅ Environment file already exists")

def check_tabpy():
    """Check if TabPy is available"""
    try:
        import tabpy
        print("✅ TabPy is available")
        return True
    except ImportError:
        print("⚠️  TabPy not found - install with: pip install tabpy")
        return False

def create_sample_config():
    """Create sample configuration files"""

    # Create sample Tableau dashboard config
    dashboard_config = {
        "dashboard_name": "Indian Traffic Accident Analysis",
        "sheets": [
            "Accident Trends",
            "State-wise Analysis", 
            "Weather Impact",
            "Time Patterns",
            "Prediction Interface"
        ],
        "kpis": [
            "Total Accidents",
            "Fatality Rate",
            "High Risk Areas",
            "Weather Impact Score"
        ]
    }

    config_file = Path("tableau_workbooks/sample_dashboard_config.json")
    config_file.parent.mkdir(exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(dashboard_config, f, indent=2)

    print("✅ Sample configuration files created")

def run_tests():
    """Run basic system tests"""
    print("Running system tests...")

    try:
        # Test data collection modules
        sys.path.append('src')
        from data_collection.accident_processor import AccidentDataProcessor

        processor = AccidentDataProcessor()
        test_df = processor.generate_synthetic_accident_data(100)

        if len(test_df) == 100:
            print("✅ Data generation test passed")
        else:
            print("❌ Data generation test failed")
            return False

        # Test model loading
        from models.ml_models import AccidentPredictionModel
        model = AccidentPredictionModel()
        print("✅ Model loading test passed")

        return True

    except Exception as e:
        print(f"❌ System tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("INDIAN ROAD TRAFFIC ACCIDENT PREDICTION SYSTEM")
    print("SETUP SCRIPT")
    print("="*60)

    # Check Python version
    if not check_python_version():
        return False

    # Create directories
    create_directories()

    # Install requirements
    if not install_requirements():
        return False

    # Setup environment
    setup_environment()

    # Check TabPy
    check_tabpy()

    # Create sample configs
    create_sample_config()

    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup continued")

    print()
    print("="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python main.py")
    print("3. For Tableau integration:")
    print("   - Install Tableau Desktop")
    print("   - Start TabPy server: tabpy")
    print("   - Follow instructions in documentation/")
    print()
    print("For detailed instructions, see README.md")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
