# Finetune-DistilBert-EndtoEnd

A project to fine-tune the DistilBERT model using the Hugging Face Transformers library. This repository covers the full pipeline: from model training, and evaluation, to deployment with FastAPI.


## Introduction

This repository provides an end-to-end pipeline for fine-tuning the [DistilBERT](https://huggingface.co/distilbert-base-multilingual-cased) model on a custom dataset and deploying it as an API using FastAPI.

Key Features:
- Fine-tune DistilBERT on text classification tasks
- API endpoint for predicting text emotion classes
- Deployed using FastAPI

## Model

We use the pre-trained `distilbert-base-multilingual-cased` model from Hugging Face. After fine-tuning, the model is deployed using FastAPI to serve predictions.

## Requirements

- Python 3.8+
- Hugging Face Transformers
- PyTorch
- FastAPI
- Uvicorn
- Pydantic
- Matplotlib

To install all the necessary dependencies, you can use the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
