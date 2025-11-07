#!/bin/bash

echo "🎙️  EMPATHY ENGINE - Quick Setup Script"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "📥 Installing dependencies (this may take 2-3 minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p templates
mkdir -p audio_output

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure index.html is in the templates/ folder"
echo "   2. Run: python app.py"
echo "   3. Open: http://localhost:5000"
echo ""
echo "🎉 Happy hacking!"