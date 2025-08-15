# Gemma 3 for Amharic Language Applications

This project leverages the Gemma 3 model (including its vision capabilities) for developing Amharic language applications, specifically focusing on chat/audio and OCR applications. It builds upon existing work with the Walia-LLM model and the amharic-dataset-mcp tools.

## 🎯 Project Goals

1. **OCR Application**: Develop an efficient OCR system for Amharic text using Gemma 3 Vision
2. **Chat/Audio Application**: Create a conversational AI that understands and responds in Amharic
3. **Efficiency**: Leverage Gemma 3's smaller size (270M parameters) for faster training and inference
4. **Integration**: Combine with existing professional datasets and tools

## 📁 Project Structure

```
.
├── gemma3_amharic_plan.md          # Detailed implementation plan
├── gemma3_amharic_demo.py          # Demonstration script
├── gemma3_amharic_requirements.txt # Python dependencies
├── gemma3_amharic_setup.sh         # Setup script
├── README.md                       # This file
├── amharic-dataset-mcp/            # MCP tools for Amharic data (separate repo)
├── amharic-llm-data/               # Walia-LLM project with professional datasets
```

## 🚀 Quick Start

1. **Run the setup script**:
   ```bash
   ./gemma3_amharic_setup.sh
   ```

2. **Activate the environment**:
   ```bash
   source gemma3_amharic_env/bin/activate
   ```

3. **Install additional dependencies**:
   ```bash
   # Install protobuf (required for Walia-LLM)
   pip install protobuf
   
   # Install Unsloth for faster training (optional)
   pip install git+https://github.com/unslothai/unsloth.git
   ```

4. **Set up HuggingFace authentication** (for Gemma models):
   ```bash
   # Get your HuggingFace token from https://huggingface.co/settings/tokens
   huggingface-cli login
   # Enter your token when prompted
   ```

5. **Run the demonstration**:
   ```bash
   python gemma3_amharic_demo.py
   ```

## 🔐 Important Authentication Notes

- **Gemma Models**: Require HuggingFace authentication with access to gated repositories
- **Walia-LLM**: Requires protobuf library and local model files
- **amharic-dataset-mcp**: May require additional API keys for data collection

## 🧠 Key Components

### 1. Gemma 3 Integration
- Language model for text generation
- Vision model for OCR applications (when available)
- Efficient training with Unsloth acceleration

### 2. Existing Tools Integration
- **amharic-dataset-mcp**: Tools for data collection and quality scoring
- **Walia-LLM**: Professional Amharic datasets (122k examples)

### 3. Amharic Language Support
- Ethiopic script processing
- Cultural context understanding
- Quality scoring for Amharic text

## 📊 Performance Advantages

Compared to the original Walia-LLM approach:
- **25x smaller**: 270M vs 7B parameters
- **6x faster training**: 4 hours vs 24 hours
- **4x less memory**: 4GB vs 16GB
- **Consumer accessible**: Runs on RTX 4090 vs requiring A100

## 🛠 Technical Implementation

### OCR Application Pipeline
```
Image Input → Gemma 3 Vision Processing → Raw Text Output → 
MCP Quality Scoring → Enhanced Amharic Text
```

### Chat Application Pipeline
```
Amharic Speech Input → ASR → Amharic Text → 
Gemma 3 Language Model → Enhanced Response → TTS → Amharic Speech Output
```

## 📈 Expected Outcomes

### OCR Application
- 85%+ character accuracy on printed Amharic text
- 75%+ word accuracy on handwritten Amharic text
- Real-time processing capabilities

### Chat/Audio Application
- Conversational AI that understands Amharic context
- Speech recognition accuracy of 80%+ for clear Amharic speech
- Natural sounding Amharic speech synthesis

## 🤝 Integration with Existing Projects

This project builds upon and integrates with:

1. **Walia-LLM Project**: Uses professional datasets and serves as performance baseline
2. **amharic-dataset-mcp**: Leverages MCP tools for data collection and quality scoring

## 📚 References

- [Gemma 3 Vision Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision.ipynb)
- [Walia-LLM Project](https://github.com/Yosef-Ali/Walia-LLM)
- [amharic-dataset-mcp](https://github.com/Yosef-Ali/amharic-dataset-mcp)

## 🌍 Impact

This project aims to democratize Amharic AI by making high-quality language models accessible to Ethiopian developers, researchers, and entrepreneurs without requiring expensive cloud resources.