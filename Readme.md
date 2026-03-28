# Fine-Tuned Llama 3.2 1B LLM model with Domain specific data

A fine-tuned Llama 3.2 1B model trained to answer questions about **BotCampus AI** (AI/ML Education Platform) and **LeadMasters AI** (Lead Generation Platform).

---

## 🔗 Model Links

| Model | Link | Size |
|-------|------|------|
| **LoRA Adapter** | [Jashu171/botcampus-leadmasters-llama3.2-1b](https://huggingface.co/Jashu171/botcampus-leadmasters-llama3.2-1b) | ~60MB |
| **GGUF (Quantized)** | [Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF](https://huggingface.co/Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF) | ~808MB |

---

## 🏗️ How I Built This

### Tech Stack

| Component | Tool |
|-----------|------|
| Base Model | `unsloth/Llama-3.2-1B-Instruct` |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) |
| Framework | Unsloth + TRL |
| Training Platform | Kaggle (Free GPU - Tesla T4) |
| Quantization | GGUF Q4_K_M (4-bit) |

### Training Details

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| Learning Rate | 2e-4 |
| Training Steps | 200 |
| Batch Size | 2 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 8 |
| Training Time | ~15 minutes |
| Dataset Size | 256 Q&A pairs |

### Step-by-Step Process

```
1. Created Dataset (256 Q&A pairs)
   ↓
2. Loaded Llama 3.2 1B with 4-bit quantization
   ↓
3. Added LoRA adapters (only 1% parameters trained)
   ↓
4. Fine-tuned on custom dataset
   ↓
5. Pushed LoRA to HuggingFace
   ↓
6. Converted to GGUF format
   ↓
7. Pushed GGUF to HuggingFace
```

---

## 🚀 Quick Start

### Option 1: Run with Ollama (Easiest)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Run model
ollama run hf.co/Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF
```

### Option 2: Run with Python

```bash
# Install dependencies
pip install llama-cpp-python huggingface_hub

# Run
python inference.py
```

### Option 3: Single Question

```bash
python inference.py -q "What is BotCampus AI?"
```

---

## 📁 Repository Structure

```
botcampus-leadmasters-finetuning/
├── README.md
├── requirements.txt
├── dataset/
│   └── botcampus_leadmasters_finetune_dataset.json
├── notebooks/
│   └── Llama_3_2_1B_Unsloth_FineTuning_Kaggle.ipynb
└── scripts/
    └── inference.py
```

---

## 📊 Dataset

The model was trained on 256 custom Q&A pairs covering:

### 🎓 BotCampus AI
- Company information
- Courses (Python, ML, Deep Learning, Cloud AI)
- Locations (Bengaluru, Dubai)
- Certifications & Workshops
- Corporate Training

### 📈 LeadMasters AI
- Platform features
- Lead generation capabilities
- Marketing automation
- Ad management (Google, Facebook, LinkedIn)
- Analytics & Insights

### Dataset Format (Alpaca)

```json
{
  "instruction": "What is BotCampus AI?",
  "input": "",
  "output": "BotCampus AI is an online learning platform specializing in AI and ML education..."
}
```

---

## 💬 Example Questions

```
❓ What is BotCampus AI?
❓ What courses does BotCampus AI offer?
❓ Where is BotCampus AI located?
❓ What is LeadMasters AI?
❓ What are the features of LeadMasters AI?
❓ How does LeadMasters AI help businesses?
```

---

## 🔧 Local API Usage

### Start Ollama Server

```bash
ollama serve
```

### Call API

```python
import requests

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "hf.co/Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF",
    "prompt": "What is BotCampus AI?",
    "stream": False
})

print(response.json()["response"])
```

### cURL

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "hf.co/Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF",
  "prompt": "What is BotCampus AI?",
  "stream": false
}'
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Training Loss (Start) | ~2.5 |
| Training Loss (End) | ~0.2 |
| Quantization | Q4_K_M (4-bit) |
| Quality Retention | ~97% |

---

## 🛠️ Reproduce Training

### 1. Open Kaggle Notebook

Upload `Llama_3_2_1B_Unsloth_FineTuning_Kaggle.ipynb` to Kaggle.

### 2. Enable GPU

Settings → Accelerator → GPU T4 x2

### 3. Upload Dataset

Add `botcampus_leadmasters_finetune_dataset.json` as input.

### 4. Run All Cells

Training takes ~15-20 minutes.

---

## 📦 Requirements

```
llama-cpp-python
huggingface_hub
unsloth
trl
peft
accelerate
bitsandbytes
```

---

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning framework
- [Meta Llama](https://llama.meta.com/) - Base model
- [Kaggle](https://kaggle.com/) - Free GPU
- [HuggingFace](https://huggingface.co/) - Model hosting

---

## 👨‍💻 Author

**Jashu171**

- HuggingFace: [Jashu171](https://huggingface.co/Jashu171)
- GitHub: [Jashu171](https://github.com/Jashu171)

---

## 📜 License

Apache 2.0

---

<div align="center">

⭐ **Star this repo if you found it helpful!** ⭐

</div>