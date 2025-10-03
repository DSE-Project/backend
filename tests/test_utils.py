"""
Utility functions for test suite to get consistent project paths
"""
import os

def get_project_paths():
    """
    Get project paths relative to the test files.
    Returns a dictionary with common project paths.
    """
    # Get the directory where this utils file is located (tests directory)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to get to the project root
    # tests_dir -> backend -> project_root  
    backend_dir = os.path.dirname(tests_dir)
    project_root = os.path.dirname(backend_dir)
    
    return {
        "project_root": project_root,
        "backend_path": backend_dir,
        "frontend_path": os.path.join(project_root, "frontend"),
        "ml_models_path": os.path.join(backend_dir, "ml_models"),
        "data_path": os.path.join(backend_dir, "data"),
        "utils_path": os.path.join(backend_dir, "utils"),
        "tests_path": tests_dir
    }

def get_model_files():
    """Get paths to ML model files"""
    paths = get_project_paths()
    return {
        "model_1m": os.path.join(paths["ml_models_path"], "1m", "model_1m.keras"),
        "model_3m": os.path.join(paths["ml_models_path"], "3m", "model_3_months.keras"),
        "scaler_1m": os.path.join(paths["ml_models_path"], "1m", "scaler_1m.pkl"),
        "scaler_3m": os.path.join(paths["ml_models_path"], "3m", "scaler_3.pkl")
    }

def get_data_files():
    """Get paths to data files"""
    paths = get_project_paths()
    return {
        "historical_1m": os.path.join(paths["data_path"], "historical_data_1m.csv"),
        "historical_3m": os.path.join(paths["data_path"], "historical_data_3m.csv"),
        "historical_6m": os.path.join(paths["data_path"], "historical_data_6m.csv")
    }

def get_config_files():
    """Get paths to configuration files"""
    paths = get_project_paths()
    return {
        "backend_env": os.path.join(paths["backend_path"], ".env"),
        "backend_env_example": os.path.join(paths["backend_path"], ".env.example"),
        "frontend_env": os.path.join(paths["frontend_path"], ".env"),
        "frontend_env_local": os.path.join(paths["frontend_path"], ".env.local"),
        "frontend_package_json": os.path.join(paths["frontend_path"], "package.json"),
        "frontend_vite_config": os.path.join(paths["frontend_path"], "vite.config.js")
    }