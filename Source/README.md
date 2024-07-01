# Cultural-Aware AI Model for Emotion Recognition

## Introduction

This repository contains the implementation of the paper "Cultural-Aware AI Model for Emotion Recognition" by Mehrdad Baradaran, Payam Zohari, Abtin Mahyar, Hossein Motamednia, Dara Rahmati, and Saeid Gorgin. The goal of the project is to develop a multimodal deep learning model that integrates cultural awareness into emotion recognition using both visual images and textual data.

## Abstract

Emotion AI aims to understand human emotions from visual or textual data. However, existing methods often ignore the influence of cultural diversity on emotional interpretation. This paper proposes a multimodal deep learning model that integrates cultural awareness into emotion recognition. Using images as the primary data source and comments from individuals across different regions as the secondary data source, the model achieves robust performance across various scenarios. This novel fusion approach bridges cultural gaps and fosters a nuanced understanding of emotions. The model is evaluated on the ArtELingo dataset, achieving an 80% recognition accuracy.


## Dataset

The model is evaluated on the ArtELingo dataset, which contains image-comment pairs with Chinese, Arabic, and English annotations. The dataset includes famous paintings classified into various art styles, annotated with emotions across multiple languages.

## Models

We experimented with various image models including VGG, ResNet50, and ViT. Ultimately, we chose ResNet50 as the image model due to its relatively lower parameter count and minimal drop in accuracy. For text processing, we used DistilBERT.  The latent features from these models were then combined and used for emotion prediction.

- The image models are located in TextImage-Modeling/Source/Image Models
- The text model is located in BreadcrumbsTextImage-Modeling/Source/Text Model
- The fusion model is located in TextImage-Modeling/Source/Fusion Mode


## Experiments and Results

The model is evaluated on the ArtELingo dataset with multilingual annotations. The results are as follows:

- Achieves an evaluation accuracy of nearly 80% on the test set, indicating robust performance in emotion recognition across different cultural contexts.

## References

For more detailed information, please refer to the full paper:

- Baradaran, M., Zohari, P., Mahyar, A., Motamednia, H., Rahmati, D., & Gorgin, S. (2024). [Link](https://ieeexplore.ieee.org/abstract/document/10491176).
