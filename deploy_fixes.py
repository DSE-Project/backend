#!/usr/bin/env python3
"""
Deployment fixes for Linux/EC2 environment
Run this script after cloning to EC2 to fix platform-specific issues
"""

import os
import sys

def fix_wkhtmltopdf_path():
    """Fix wkhtmltopdf path for Linux environment"""
    main_py_path = "main.py"
    
    if not os.path.exists(main_py_path):
        print("❌ main.py not found")
        return False
    
    # Read the file
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Replace Windows path with Linux path
    old_path = 'wkhtmltopdf_path = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"'
    new_path = 'wkhtmltopdf_path = "/usr/bin/wkhtmltopdf"'
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        
        # Write back to file
        with open(main_py_path, 'w') as f:
            f.write(content)
        
        print("✅ Fixed wkhtmltopdf path for Linux")
        return True
    else:
        print("⚠️ wkhtmltopdf path pattern not found or already fixed")
        return False

def create_env_template():
    """Create .env template file"""
    env_template = """# Environment Variables for RecessionScope Backend
# Copy this file to .env and fill in your actual values

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# GROQ API (for sentiment analysis)
GROQ_API_KEY=your_groq_api_key_here

# FRED API (Federal Reserve Economic Data)
FRED_API_KEY=your_fred_api_key_here

# Reddit API (for sentiment analysis)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=RecessionScope/1.0

# Application Settings
ENVIRONMENT=production
DEBUG=false
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.example template")
    print("📝 Please copy to .env and fill in your actual values:")
    print("   cp .env.example .env")
    print("   nano .env")

if __name__ == "__main__":
    print("🔧 Running deployment fixes...")
    fix_wkhtmltopdf_path()
    create_env_template()
    print("✅ Deployment fixes complete!")