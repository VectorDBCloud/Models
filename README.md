![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)

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
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contribution and Feedback](#contribution-and-feedback)
- [Related Resources](#related-resources)
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
  *Last Updated*: 2023-05-15

- **SBERT (Sentence-BERT)**
  *Description*: An extension of BERT optimized for generating high-quality sentence embeddings, ideal for clustering and semantic search.
  *Link*: [SBERT on Hugging Face](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens)
  *Last Updated*: 2023-06-01

### Image Embeddings

- **ResNet (Residual Networks)**
  *Description*: A popular deep learning model used for image recognition tasks, suitable for generating image embeddings.
  *Link*: [ResNet on Hugging Face](https://huggingface.co/microsoft/resnet-50)
  *Last Updated*: 2023-04-20

- **CLIP (Contrastive Language–Image Pretraining)**
  *Description*: A model capable of understanding images and text together, useful for multimodal applications.
  *Link*: [CLIP on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
  *Last Updated*: 2023-05-30

### Multimodal Models

- **VisualBERT**
  *Description*: Combines visual and textual information, great for tasks that require understanding both image and text inputs.
  *Link*: [VisualBERT on Hugging Face](https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre)
  *Last Updated*: 2023-03-10
 
- **VilBERT**
  *Description*: A model designed for tasks that require joint understanding of vision and language.
  *Link*: [VilBERT on Hugging Face](https://huggingface.co/facebook/vilbert-multi-task)
  *Last Updated*: 2023-04-05

## Usage

To use these models with vector databases:

1. Select a model suitable for your task.
2. Install the required libraries:
   ```
   pip install transformers torch
   ```
3. Load the model and generate embeddings:
   ```python
   from transformers import AutoModel, AutoTokenizer
   import torch

   # Load model and tokenizer
   model_name = "bert-base-uncased"
   model = AutoModel.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   # Generate embeddings
   text = "Example sentence for embedding"
   inputs = tokenizer(text, return_tensors="pt")
   with torch.no_grad():
       embeddings = model(**inputs).last_hidden_state.mean(dim=1)

   # Use embeddings with your vector database
   ```
4. Store and query these embeddings in your chosen vector database.

For specific integration examples, check the documentation of your vector database system.

## Best Practices

1. Choose the appropriate model for your specific use case and data type.
2. Fine-tune models on your domain-specific data when possible for better performance.
3. Regularly update your models to benefit from the latest improvements.
4. Implement proper error handling and logging in your embedding generation pipeline.
5. Consider the computational resources required for each model, especially for large-scale applications.

## Troubleshooting

- **Issue**: Model not found when loading
  **Solution**: Ensure you have an active internet connection and the model name is correct.

- **Issue**: Out of memory errors
  **Solution**: Try using a smaller batch size or a more memory-efficient model.

- **Issue**: Slow embedding generation
  **Solution**: Consider using GPU acceleration or a more lightweight model for faster processing.

## Contribution and Feedback

We encourage contributions from the community! If you have a model that works well with vector databases or enhancements to existing models, please follow these steps:

1. Fork the repository.
2. Create a new branch for your contribution.
3. Add your model or make your changes. Include comprehensive documentation, including installation instructions, usage examples, and any dependencies.
4. Submit a pull request with a clear description of your contribution.

Please ensure all models are properly licensed and attributed, and clearly state the purpose and potential applications of the model.

For any issues or suggestions, please use the issue tracker.

## Related Resources

- [Vector Database Cloud Documentation](https://docs.vectordbcloud.com)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Embeddings Repository](https://github.com/VectorDBCloud/Embeddings)
- [Tutorials Repository](https://github.com/VectorDBCloud/tutorials)

## Code of Conduct

We adhere to the [Vector Database Cloud Code of Conduct](https://github.com/VectorDBCloud/Community/blob/main/CODE_OF_CONDUCT.md). Please respect these guidelines when contributing to or using this repository.


## License

This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).

Copyright (c) 2024 Vector Database Cloud

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution — You must give appropriate credit to Vector Database Cloud, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests Vector Database Cloud endorses you or your use.

Additionally, we require that any use of this guide includes visible attribution to Vector Database Cloud. This attribution should be in the form of "Based on Models curated by Vector Database Cloud", along with a link to https://vectordbcloud.com, in any public-facing applications, documentation, or redistributions of this guide.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For the full license text, visit: https://creativecommons.org/licenses/by/4.0/legalcode



## Disclaimer

The information and resources provided in this community repository are for general informational purposes only. While we strive to keep the information up-to-date and correct, we make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the information, products, services, or related graphics contained in this repository for any purpose. Any reliance you place on such information is therefore strictly at your own risk.

Vector Database Cloud configurations may vary, and it's essential to consult the official documentation before implementing any solutions or suggestions found in this community repository. Always follow best practices for security and performance when working with databases and cloud services.

The content in this repository may change without notice. Users are responsible for ensuring they are using the most current version of any information or code provided.

This disclaimer applies to Vector Database Cloud, its contributors, and any third parties involved in creating, producing, or delivering the content in this repository.

The use of any information or code in this repository may carry inherent risks, including but not limited to data loss, system failures, or security vulnerabilities. Users should thoroughly test and validate any implementations in a safe environment before deploying to production systems.

For complex implementations or critical systems, we strongly recommend seeking advice from qualified professionals or consulting services.

By using this repository, you acknowledge and agree to this disclaimer. If you do not agree with any part of this disclaimer, please do not use the information or resources provided in this repository.
