"""
BotCampus AI & LeadMasters AI - Inference Script
Model: Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF
"""

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import argparse

# ============================================
# DOWNLOAD & LOAD MODEL
# ============================================
def load_model():
    print("📥 Downloading model...")
    
    model_path = hf_hub_download(
        repo_id="Jashu171/botcampus-leadmasters-llama3.2-1b-GGUF",
        filename="llama-3.2-1b-instruct.Q4_K_M.gguf"
    )
    
    print("🔄 Loading model...")
    llm = Llama(model_path=model_path, n_ctx=2048)
    
    print("✅ Model ready!")
    return llm

# ============================================
# ASK FUNCTION
# ============================================
def ask(llm, question):
    prompt = f"""Below is an instruction that describes a task.

### Instruction:
{question}

### Response:
"""
    output = llm(prompt, max_tokens=256, stop=["### Instruction:"])
    return output["choices"][0]["text"].strip()

# ============================================
# INTERACTIVE CHAT
# ============================================
def chat(llm):
    print("\n" + "=" * 50)
    print("🤖 BotCampus & LeadMasters AI Chatbot")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("❓ You: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question.strip():
            continue
        
        answer = ask(llm, question)
        print(f"🤖 Bot: {answer}\n")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BotCampus & LeadMasters AI")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    args = parser.parse_args()
    
    llm = load_model()
    
    if args.question:
        # Single question mode
        print(ask(llm, args.question))
    else:
        # Interactive chat mode
        chat(llm)