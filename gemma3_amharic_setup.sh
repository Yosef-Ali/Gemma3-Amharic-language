#!/bin/bash
# Gemma 3 for Amharic Applications - Setup Script

echo "🚀 Setting up Gemma 3 for Amharic Applications"
echo "================================================"

# Check if we're on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "💻 Running on macOS"
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "⚠️  Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Running on Linux"
else
    echo "❓ Unsupported OS: $OSTYPE"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv gemma3_amharic_env

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source gemma3_amharic_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core requirements
echo "📦 Installing core requirements..."
pip install -r gemma3_amharic_requirements.txt

# Install additional packages that might be needed
echo "➕ Installing additional packages..."
pip install tqdm requests

# Test installation
echo "🧪 Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Check if we have access to the Amharic dataset tools
if [ -d "amharic-dataset-mcp" ]; then
    echo "🔗 Found amharic-dataset-mcp directory"
    # Install in development mode
    pip install -e amharic-dataset-mcp
else
    echo "⚠️  amharic-dataset-mcp not found in current directory"
    echo "   Please ensure you have cloned the amharic-dataset-mcp repository"
fi

# Check if we have the Amharic LLM data
if [ -d "amharic-llm-data" ]; then
    echo "🔗 Found amharic-llm-data directory"
else
    echo "⚠️  amharic-llm-data not found in current directory"
fi

echo ""
echo "✅ Setup completed!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source gemma3_amharic_env/bin/activate"
echo ""
echo "To test the Amharic applications, run:"
echo "   python gemma3_amharic_demo.py"
echo ""