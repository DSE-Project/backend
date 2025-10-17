#!/bin/bash
# Update script for RecessionScope Backend

set -e

echo "🔄 Updating RecessionScope Backend..."

cd /var/www/recessionscope/backend

echo "📥 Pulling latest changes..."
git pull origin main  # or your branch name

echo "📦 Updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt --upgrade

echo "🎭 Updating Playwright..."
playwright install chromium

echo "🔄 Restarting services..."
sudo systemctl restart recessionscope
sudo systemctl restart nginx

echo "🧪 Health check..."
sleep 5
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Update successful! Application is healthy."
else
    echo "❌ Health check failed!"
    exit 1
fi

echo "✅ Update complete!"