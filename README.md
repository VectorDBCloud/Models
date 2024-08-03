![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

# Models

Welcome to the Vector Database Cloud Models repository! This repository curates a list of Hugging Face models optimized for use with vector databases such as pgvector, Milvus, Qdrant, and ChromaDB. These models enhance functionalities like semantic search, classification, and other machine learning applications.

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Models](#models)
  - [Text Embeddings](#text-embeddings)
  - [Image Embeddings](#image-embeddings)
  - [Multimodal Models](#multimodal-models)
- [Usage](#usage)
- [Contribution and Feedback](#contribution-and-feedback)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Disclaimer](#disclaimer)

## About

This repository serves as a centralized resource for finding and sharing machine learning models that integrate seamlessly with vector databases. The curated models are selected based on their performance and compatibility with vector database technologies, enabling developers to enhance their applications with advanced AI capabilities.

## Prerequisites

- Python 3.7+
- Knowledge of machine learning and vector databases
- Familiarity with Hugging Face's transformers library
- Access to vector database systems (e.g., pgvector, Milvus, Qdrant, ChromaDB)

## Models

### Text Embeddings

- **BERT (Bidirectional Encoder Representations from Transformers)**  
  *Description*: A transformer-based model designed for various NLP tasks, including semantic search and question answering.  
  *Link*: [BERT on Hugging Face](https://huggingface.co/bert-base-uncased)

- **SBERT (Sentence-BERT)**  
  *Description*: An extension of BERT optimized for generating high-quality sentence embeddings, ideal for clustering and semantic search.  
  *Link*: [SBERT on Hugging Face](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)

### Image Embeddings

- **ResNet (Residual Networks)**  
  *Description*: A popular deep learning model used for image recognition tasks, suitable for generating image embeddings.  
  *Link*: [ResNet on Hugging Face](https://huggingface.co/microsoft/resnet-50)

- **CLIP (Contrastive Language–Image Pretraining)**  
  *Description*: A model capable of understanding images and text together, useful for multimodal applications.  
  *Link*: [CLIP on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)

### Multimodal Models

- **VisualBERT**  
  *Description*: Combines visual and textual information, great for tasks that require understanding both image and text inputs.  
  *Link*: [VisualBERT on Hugging Face](https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre)

- **VilBERT**  
  *Description*: A model designed for tasks that require joint understanding of vision and language.  
  *Link*: [VilBERT on Hugging Face](https://huggingface.co/facebook/vilbert-multi-task)

## Usage

To use these models with vector databases:

1. Select a model suitable for your task.
2. Follow the installation instructions provided in the model's Hugging Face page.
3. Use the model to generate embeddings for your data.
4. Store and query these embeddings in your chosen vector database.

For specific integration examples, check the documentation of your vector database system.

## Contribution and Feedback

We encourage contributions from the community! If you have a model that works well with vector databases or enhancements to existing models, please follow these steps:

1. Fork the repository.
2. Create a new branch for your contribution.
3. Add your model or make your changes. Include comprehensive documentation, including installation instructions, usage examples, and any dependencies.
4. Submit a pull request with a clear description of your contribution.

Please ensure all models are properly licensed and attributed, and clearly state the purpose and potential applications of the model.

For any issues or suggestions, please use the issue tracker.


## Code of Conduct

We adhere to the [Vector Database Cloud Code of Conduct](https://github.com/VectorDBCloud/Community/blob/main/CODE_OF_CONDUCT.md). Please respect these guidelines when contributing to or using this repository.


## Disclaimer

The models listed in this repository are third-party creations and are subject to their respective licenses and terms of use. While we strive to ensure the quality and suitability of these models for use with vector databases, Vector Database Cloud is not responsible for the performance, accuracy, or any consequences resulting from the use of these models.

Users should carefully review the license and usage terms of each model before incorporating them into their projects. It is the responsibility of the user to ensure compliance with all applicable licenses and to verify the suitability of the model for their specific use case.

The integration examples and usage suggestions provided in this repository are for illustrative purposes only and may need to be adapted to work with specific vector database implementations or versions. Users should thoroughly test and validate the models and their integrations in a non-production environment before deploying them in any critical or production systems.

Vector Database Cloud does not provide support for the individual models listed here. For issues related to specific models, please refer to their respective documentation or support channels.

Always follow best practices for security, data privacy, and model deployment when working with machine learning models and vector databases, especially when handling sensitive data or deploying in production environments.
