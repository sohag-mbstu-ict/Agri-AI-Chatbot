# Agri AI Chatbot

## RAG + Fine-Tuned LLM + Tool-Augmented Intelligence

An intelligent agriculture-focused chatbot that combines Retrieval-Augmented Generation (RAG), a fine-tuned Gemma (SFT) model, and real-time web/tool integration to deliver accurate, context-aware answers for farmers and agri-users.

## Features
### Hybrid AI Architecture
  - Combines RAG + Fine-Tuned LLM (Gemma SFT)
  - Enables both fact-based retrieval and contextual generation

### Domain-Specific RAG Dataset
#### Structured agricultural knowledge with metadata:
  - Crop
  - Disease
  - Pest
  - Symptoms
  - Remedies
  - Improves retrieval precision and relevance

## Fine-Tuned LLM (Gemma SFT)
  - Trained on agricultural Q&A dataset
  - Reduces hallucination and improves domain understanding

## Tool-Augmented Intelligence
  - Real-time knowledge from:
  - DuckDuckGo Search
  - Tavily
  - Wikipedia
  - Arxiv
  - Enhances responses beyond static data
## Smart Query Experience
  - Real-time word suggestions (autocomplete)
  - Query suggestions based on intent and usage patterns

## Multilingual Support
  - Supports both English and Bangla queries
## Interactive UI
  - Built with django, javascript, html css
  - Clean and user-friendly interface

## Tech Stack
  - LLM: Gemma (Fine-Tuned with SFT)
  - Frameworks: LangChain, LangGraph
  - Search Tools: Tavily, DuckDuckGo
  - Data Sources: Wikipedia, Arxiv
  - Embeddings: BGE / HuggingFace
  - Vector Store: FAISS / Chroma
