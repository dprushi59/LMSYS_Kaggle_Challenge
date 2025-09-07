# 🤖 Chatbot Arena Preference Prediction  

This repository contains my solution for the **Kaggle competition: "Predict Which LLM Response Users Prefer"** ([Chatbot Arena Preference Prediction](https://www.kaggle.com/competitions/lmsys-chatbot-arena)).  
The task is to predict **which response users will prefer** in head-to-head battles between large language models (LLMs).  

---

## 📌 Competition Overview  

Large language models are rapidly entering our lives, but ensuring their outputs **align with human preferences** is critical.  
This competition provides prompts and two responses from different LLMs. Users then vote for the response they prefer (Model A, Model B, or Tie).  

- **Train Data**: ~55K rows of Chatbot Arena interactions  
- **Test Data**: ~25K rows (hidden ground truth)  
- **Target**: Multi-class classification → `winner_model_a`, `winner_model_b`, `winner_tie`  

This aligns with **reward modeling** and **reinforcement learning from human feedback (RLHF)**.  

---

## ⚙️ Approach  

### 🔹 1. Prompt-based Preference Modeling  
- Designed a **discriminator prompt** comparing two responses across **5 criteria**:  
  * Relevance  
  * Accuracy  
  * Clarity  
  * Logical Flow  
  * Responsiveness  
- The model outputs votes → `A`, `B`, `AB`, or `NA`.  
- Aggregated votes are converted to **softmax probabilities** for (A, B, Tie).  

### 🔹 2. Synthetic Data Augmentation  
- Used **LLaMA-3.1-405B** to curate a **20k+ sample synthetic dataset** with richer annotations.  
- Helped fine-tune smaller models effectively.  

### 🔹 3. Model Training  
- Fine-tuned **Phi-3-Mini-4k** and **LLaMA-3.1-8B** with **PEFT (LoRA adapters)**.  
- Framework: **vLLM** for inference + **LangChain text splitters** for handling long prompts.  

### 🔹 4. Inference Optimization  
- Enabled **KV-caching** for faster repeated inference.  
- Implemented **multi-threading across dual GPUs**.  
- Applied **CUDA graph capture** to stabilize performance.  

---

## 📊 Results  

- Improved **log loss: 11.892 → 0.949**  
- Efficient inference speed: **~2000 tokens/sec** throughput  
- Achieved competitive leaderboard performance  

---

## 🛠 Tech Stack  

- **Models**: Phi-3-Mini-4k, LLaMA-3.1-8B, LLaMA-3.1-405B  
- **Frameworks**: [vLLM](https://github.com/vllm-project/vllm), PyTorch, LangChain  
- **Training**: PEFT, LoRA  
- **Environment**: Kaggle Notebooks, Dual NVIDIA GPUs  

---
