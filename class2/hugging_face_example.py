from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------
# 📦 Device Selection: CUDA > MPS > CPU
# ------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA (GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🟡 Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("🔴 Using CPU")

# ------------------------------------------
# 🧠 Load Model from Hugging Face
# ------------------------------------------
model_name = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32  # avoid FP16 on CPU
).to(device)

# ------------------------------------------
# 📝 Prompt + Inference
# ------------------------------------------
prompt = "The Eiffel Tower is located in"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("📝 Generated Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
