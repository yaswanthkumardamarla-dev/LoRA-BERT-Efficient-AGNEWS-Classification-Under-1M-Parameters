# 📰 Text Classification under 1 million parameters: Fine-Tuning RoBERTa with LoRA on AGNEWS

**CS-GY 6953 / ECE-GY 7123 - Deep Learning Project 2 (Spring 2025)**  
**New York University (NYU) Tandon School of Engineering**

---

## 📌 Overview

This repository contains an efficient fine-tuning implementation of the `roberta-base` model using **Low-Rank Adaptation (LoRA)** for text classification on the **AGNEWS** dataset. The goal is to achieve **high accuracy** with **fewer than 1 million trainable parameters**.

---

## 🧠 Model Architecture

- **Base Model**: [`roberta-base`](https://huggingface.co/roberta-base)
- **LoRA Applied To**:
  - `self-attention.query`
  - `self-attention.value`
  - `classifier.dense`
- **LoRA Configuration**:
  - `rank (r)`: 1
  - `alpha`: 2
  - `dropout`: 0.5
  - `bias`: none
- **Frozen Parameters**: All pre-trained RoBERTa parameters
- **Trainable Components**: LoRA adapters + classifier head
- **Classifier Head**: Dense → Output layer (4 classes)

### 🔢 Trainable Parameters
- **Total**: 632,068  
- **Percentage**: 0.5045% of the full model (125M parameters)

---

## 🛠️ Training Methodology

### 🧰 Framework & Tools
- **Libraries**: PyTorch, Hugging Face Transformers, [PEFT](https://github.com/huggingface/peft)
- **Hardware**: GPU acceleration with **mixed precision** (fp16=True)

### 🧼 Data Preprocessing
- **Dataset**: [AGNEWS](https://huggingface.co/datasets/ag_news) (4 classes of news articles)
- **Tokenizer**: `RobertaTokenizer`
  - Max length: 256 tokens
  - Truncation and dynamic padding
- **Split**: 90% Train / 10% Validation

### ⚙️ Optimization
- **Optimizer**: Adam
- **Learning Rate**: 7e-4
- **Weight Decay**: 0.003

### 📊 Training Configuration
- **Epochs**: 3
- **Batch Size**: 32 (train), 64 (validation)
- **Early Stopping**: Enabled (patience = 3)
- **Mixed Precision Training**: Enabled

---

## 📈 Results & Performance

| Metric                  | Value         |
|------------------------|---------------|
| ✅ Validation Accuracy | **94.44%**    |
| 🔥 Training Loss       | 0.156         |
| 🧮 Trainable Params    | 632,068       |
| ⏱ Runtime              | ~12 minutes   |
| ⚡ Throughput (samples/sec) | 442.1   |
| ⚡ Throughput (steps/sec)   | 13.82   |

---

## 🔍 Evaluation & Inference

- Predictions on the test set were generated using the `DL_Project_2_final.ipynb` notebook.
- Final predictions saved in: `project2_best_submission.csv` (CSV format).

---

## 📂 Repository Structure

```bash
.
├── DL_Project_2_final.ipynb                # Training and fine-tuning script with LoRA
├── README.md                               # Project documentation
└── project2_best_submission.csv            # Final test predictions
