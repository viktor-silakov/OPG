#!/bin/bash

echo "🐟 Setting up fs-python environment..."

# Check for Poetry
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "✅ Poetry installed"
    echo "⚠️ You may need to restart the terminal or run: source ~/.bashrc"
fi

# Go to fs-python directory
cd fs-python || exit 1

echo "📦 Installing dependencies with Poetry..."
poetry install

echo "🐟 Installing Fish Speech..."
poetry run pip install -e ./fish-speech

echo "📁 Creating necessary directories..."
mkdir -p voices
mkdir -p output
mkdir -p cache

echo "🎭 Creating example voice files..."
cat > voices/example.txt << 'EOF'
This is an example text for a reference voice. Record audio with this text and create a .npy file with the command: poetry run python cli_tts.py --create-reference audio.wav voices/example.npy
EOF

echo "✅ fs-python environment setup complete!"
echo "💡 You can now run: npm start"

echo "📋 Next steps:"
echo "1. For creating reference voices:"
echo "   cd fs-python && poetry run python cli_tts.py --create-reference audio.wav voices/speaker_name.npy"
echo ""
echo "2. For testing generation:"
echo "   cd fs-python && poetry run python cli_tts.py 'Hello world' --play"
echo ""
echo "3. For running the main process:"
echo "   npm run process" 