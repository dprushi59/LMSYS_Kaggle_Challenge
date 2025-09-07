Chatbot Arena Preference Prediction

This repository contains my solution for the Kaggle competition: "Predict Which LLM Response Users Prefer" (Chatbot Arena Preference Prediction
).
The goal of the competition was to predict which response users preferred in head-to-head battles between large language models (LLMs).

📌 Competition Overview

Large language models are becoming integral to our daily lives, but ensuring their outputs align with human preferences remains a challenge.
In this competition, we are given prompts and responses from two different LLMs, along with user choices (Model A, Model B, or Tie). The task is to build a model that can predict these preferences.

Train Data: ~55K rows of Chatbot Arena interactions

Test Data: ~25K rows (hidden ground truth)

Target: Multi-label classification → winner_model_a, winner_model_b, winner_tie

This task aligns with reward models / preference models in RLHF (Reinforcement Learning from Human Feedback).

⚙️ Approach
1. Prompt-based Preference Modeling

Instead of directly training on structured features, I framed this as a discriminator task:

Designed a prompt template that compares two responses across 5 criteria (Relevance, Accuracy, Clarity, Logical Flow, Responsiveness).

The model outputs A / B / AB / NA labels for each criterion.

These votes are aggregated and converted into softmax probabilities for (Model A, Model B, Tie).

2. Synthetic Data Augmentation

Leveraged LLaMA-3.1-405B to generate a 20k+ sample synthetic dataset with richer preference annotations.

This helped fine-tune smaller models effectively.

3. Model Training

Fine-tuned Phi-3-Mini-4k and LLaMA-3.1-8B using PEFT (Parameter-Efficient Fine-Tuning).

Used vLLM for fast inference and LangChain text splitters for handling long prompts.

4. Inference Optimization

Enabled KV caching to reduce repeated computation.

Used multi-threading across dual GPUs to parallelize evaluation.

Applied CUDA Graph capture to stabilize performance.

📊 Results

Optimized log loss from 11.892 → 0.949

Achieved competitive leaderboard performance

Efficient inference speed with ~2000 tokens/sec input throughput

🛠 Tech Stack

Models: Phi-3-Mini-4k, LLaMA-3.1-8B, LLaMA-3.1-405B

Frameworks: vLLM
, PyTorch, LangChain

Training: PEFT, LoRA adapters

Environment: Kaggle Notebooks, dual NVIDIA GPUs
