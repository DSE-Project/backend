#!/bin/bash
# RecessionScope Backend Deployment Script for EC2
# Run this script after cloning the repository to EC2

set -e  # Exit on any error

echo "🚀 Starting RecessionScope Backend Deployment..."

# Check if running as ubuntu user
if [ "$USER" != "ubuntu" ]; then
    echo "❌ Please run this script as ubuntu user"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the backend directory"
    exit 1
fi

echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "🎭 Installing Playwright browsers..."
playwright install chromium
playwright install-deps

echo "🔧 Applying deployment fixes..."
python deploy_fixes.py

echo "📝 Environment setup..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your actual API keys:"
    echo "   nano .env"
    echo ""
    echo "Required variables:"
    echo "- SUPABASE_URL"
    echo "- SUPABASE_ANON_KEY" 
    echo "- GROQ_API_KEY"
    echo "- FRED_API_KEY"
    echo ""
    read -p "Press Enter after you've configured .env file..."
fi

echo "🧪 Testing application startup..."
python main.py &
APP_PID=$!
sleep 10

# Test if app is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application test successful!"
    kill $APP_PID
else
    echo "❌ Application test failed!"
    kill $APP_PID 2>/dev/null || true
    exit 1
fi

echo "🎉 Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Configure Nginx reverse proxy"
echo "2. Set up SSL certificate"
echo "3. Configure systemd service for auto-restart"
echo "4. Set up monitoring"