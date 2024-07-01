# Artelingo Dataset: Multimodal Emotion Recognition

This repository contains code and resources for training and evaluating multimodal emotion recognition models using the Artelingo dataset. The project involves separate text and image models, which are later fused for improved emotion recognition performance.

## Project Structure

The project is organized into the following directories:

- `TextImage-Modeling/Source/Text Model`: Contains the code for training the text-based emotion recognition model.
- `TextImage-Modeling/Source/Image Models`: Contains the code for training image-based emotion recognition models using ResNet50, VGG, and ViT.
- `TextImage-Modeling/Source/Fusion Model`: Contains the code for combining the text and image models into a unified multimodal model.
- `TextImage-Modeling/Weights`: Contains a text file with links to pretrained model weights.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.8 or higher
- PyTorch
- torchvision
- Transformers (Hugging Face)
- OpenCV
- PIL
- Other dependencies listed in `requirements.txt`

### Pretrained Model Weights

Pretrained weights for the models (ResNet50, DistilBERT, and the fusion model) are available in the following file:

- ``` Weights/Weights_Kaggle_link.txt```

You can download these weights and use them directly without the need to train the models from scratch.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/artelingo-multimodal-emotion-recognition.git
    cd artelingo-multimodal-emotion-recognition
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Training the Models

### Text Model

Navigate to the `BreadcrumbsTextImage-Modeling/Source/Text Model` directory and run the Jupyter notebook `ArtelingoTextProcessing.ipynb` to train the text model.

### Image Models

Navigate to the `TextImage-Modeling/Source/Image Models` directory and run the respective Python scripts to train the image models:

```bash
python emotion_recognition_multilingual_model_resnet50.py  # ResNet50
python emotion_recognition_multilingual_model_vgg.py
python emotion_recognition_multilingual_model_vit.py
```
### Fusion Model
After training the text and image models, navigate to the `TextImage-Modeling/Source/Fusion Model` directory and run the fusion model script to combine the latent representations and perform emotion recognition:
```bash
python fusion_model.py
```

### Citation

If you use this repository in your research or work, please cite the following paper:

```scss
@inproceedings{baradaran2024cultural,
  title={Cultural-Aware AI Model for Emotion Recognition},
  author={Baradaran, Mehrdad and Zohari, Payam and Mahyar, Abtin and Motamednia, Hossein and Rahmati, Dara and Gorgin, Saeid},
  booktitle={2024 13th Iranian/3rd International Machine Vision and Image Processing Conference (MVIP)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

This citation acknowledges the paper that introduced the cultural-aware AI model for emotion recognition, aligning with the objectives of the Artelingo dataset project.
