# Models

Welcome to the Vector Database Cloud Models repository! This repository curates a list of Hugging Face models optimized for use with vector databases such as pgvector, Milvus, Qdrant, and ChromaDB. These models enhance functionalities like semantic search, classification, and other machine learning applications.

## Table of Contents

- [About](#about)
- [How to Contribute](#how-to-contribute)
- [Models](#models)
  - [Text Embeddings](#text-embeddings)
  - [Image Embeddings](#image-embeddings)
  - [Multimodal Models](#multimodal-models)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

## About

This repository serves as a centralized resource for finding and sharing machine learning models that integrate seamlessly with vector databases. The curated models are selected based on their performance and compatibility with vector database technologies, enabling developers to enhance their applications with advanced AI capabilities.

## How to Contribute

We encourage contributions from the community! If you have a model that works well with vector databases or enhancements to existing models, please contribute by following these steps:

1. **Fork the Repository**: Fork this repository to your GitHub account.
2. **Add Your Model**: Create a new directory for your model. Include any scripts, model weights, and a README with instructions and use cases.
3. **Submit a Pull Request**: After adding your model, submit a pull request for review and inclusion in the repository.

### Contribution Guidelines

- Ensure that all models are properly licensed and attributed.
- Include comprehensive documentation, including installation instructions, usage examples, and any dependencies.
- Clearly state the purpose and potential applications of the model.

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

## Code of Conduct

We adhere to the [Vector Database Cloud Code of Conduct](https://github.com/VectorDBCloud/Community/blob/main/CODE_OF_CONDUCT.md). Please respect these guidelines when contributing to or using this repository.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
