#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer
import pandas as pd
from PIL import Image
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import csv
import tqdm


# In[2]:


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


image_model = models.resnet50(pretrained=False)

# Modify the top classification head
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 768),
)


# In[4]:


# Load the pretrained weights for the first linear layer
checkpoint = torch.load('/kaggle/input/resnet50-weights-multilingual/best_model_resnset50.pth')
model_dict = image_model.state_dict()
checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('custom_head.0')}  # Only load weights for the first linear layer
model_dict.update(checkpoint)
image_model.load_state_dict(model_dict)

# Set the model to evaluation mode
image_model.eval()


# In[5]:


# Define a custom DistilBERT model that stops at a specific layer
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]

        return pooler  # Exclude the last layer to stop before pooler


# In[6]:


# Initialize the custom feature extractor model
text_model = DistillBERTClass()

# Load the pretrained weights into the feature extractor model
model_weights_path = "/kaggle/input/distilbert-weights-multilingual/model_txt.pth"
loaded_model_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))

# Remove unnecessary keys from the loaded state_dict
# (assuming "pre_classifier" and "classifier" layers are not needed)
filtered_state_dict = {k: v for k, v in loaded_model_state_dict.state_dict().items() if "pre_classifier" not in k and "classifier" not in k}

# Load the filtered state_dict into the feature extractor model
text_model.load_state_dict(filtered_state_dict, strict=False)

# Set the feature extractor model in evaluation mode
text_model.eval()


# In[7]:


# Define the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")


# In[8]:


df1 = pd.read_csv('/kaggle/input/dev-test-artelingo/emo_pred_challenge_dev.csv')


# In[9]:


df1.head()


# In[10]:


df1['image_file'] = '/kaggle/input/wikiart/' + df1['art_style'] + '/' + df1['painting'] + '.jpg'


# In[11]:


df1.head()


# In[12]:


from difflib import SequenceMatcher


# In[13]:


def preprocess_dataset(df, text_model, image_model, transform):
    image_model = image_model.to(device)
    text_model = text_model.to(device)
    image_id = []
    data = []
    nist = []
    # Iterate through the dataframe and drop rows with non-existing image paths
    indices_to_drop = []
    for idx, row in tqdm.tqdm(df.iterrows()):
        image_id.append(row['id'])
        image_path = row['image_file'] 
        if not os.path.exists(image_path):
            incorrect_path = image_path
            max_similarity = 0
            most_similar_name = None
            # Path to the root directory where your dataset is located
            root_directory = "/kaggle/input/wikiart/"     
            # Extract the common prefix
            common_prefix = incorrect_path.split(root_directory)[1].split("/")[0]

            # Iterate through the files in the directory
            for dirpath, dirnames, filenames in os.walk(os.path.join(root_directory, common_prefix)):
                for filename in filenames:
                # Extract just the image name from the path
                    dataset_path = os.path.join(dirpath, filename)

                    # Calculate the similarity score using Levenshtein distance
                    similarity = SequenceMatcher(None, os.path.basename(incorrect_path), os.path.basename(dataset_path)).ratio()
                    # If the current name is more similar, update the result
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_name = os.path.basename(dataset_path)
                        
            corrected_path = incorrect_path.replace(os.path.basename(incorrect_path), most_similar_name)
            img_name = corrected_path
            # Load and preprocess the image
            image = Image.open(img_name)
            if transform:
                image = transform(image)

            utterance = row['utterance']
            input_ids = tokenizer.encode(utterance, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
            attention_mask = (input_ids != 0).float()

            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image_features = image_model(image.unsqueeze(0))
            text_features = text_model(input_ids.squeeze(1), attention_mask.squeeze(1))
            image_features = image_features.cpu().detach().numpy()
            text_features = text_features.cpu().detach().numpy()
            data.append((image_features, text_features))
        else:
            img_name = row['image_file']
            # Load and preprocess the image
            image = Image.open(img_name)
            if transform:
                image = transform(image)

            utterance = row['utterance']
            input_ids = tokenizer.encode(utterance, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
            attention_mask = (input_ids != 0).float()

            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image_features = image_model(image.unsqueeze(0))
            text_features = text_model(input_ids.squeeze(1), attention_mask.squeeze(1))
            image_features = image_features.cpu().detach().numpy()
            text_features = text_features.cpu().detach().numpy()
            data.append((image_features, text_features))

    df = df.drop(indices_to_drop)
    df.reset_index(drop=True, inplace=True)
    return df, data, image_id


# In[14]:


# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[15]:


df, data, image_id = preprocess_dataset(df1, text_model, image_model, transform)


# In[ ]:


len(df1), len(df)


# In[ ]:


df1['image_file'][64]


# In[ ]:


df['image_file'][64]


# In[ ]:


data[0][0].shape, data[0][1].shape


# In[ ]:


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, data, image_id):
        self.df = df
        self.data = data
        self.image_id = image_id
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_id[idx]
        img_name = self.df.loc[idx, 'image_file']

        # Load and preprocess the image
        image_features = torch.from_numpy(self.data[idx][0])
        text_features = torch.from_numpy(self.data[idx][1])

        return image_features.to(device), text_features.to(device)


# In[ ]:


dataset = ImageTextDataset(df, data, image_id)
dataloader = DataLoader(dataset, batch_size=256)


# In[ ]:


class FusionModel(nn.Module):
    def __init__(self, fusion_dim, num_classes):
        super(FusionModel, self).__init__()
        self.fusion_dim = fusion_dim

        # Define the linear transformation
        self.linear_2_to_1 = nn.Linear(2, 1)

        # Define a more complex classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Adding more layers
            nn.BatchNorm1d(256),  # Batch normalization layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Softmax activation for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, text_features):
        # Concatenate image and text features
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = combined_features.transpose(1, 2)
        # Apply the linear transformation
        fused_latent = self.linear_2_to_1(combined_features)
        # Apply the complex classifier
        fused_latent = fused_latent.squeeze()
        
        output = self.classifier(fused_latent)

        # Apply softmax activation for classification
        output = self.softmax(output)

        return output.to(device)


# In[ ]:


# Initialize the FusionModel for emotion recognition
num_classes = 9
fusion_dim = 768 * 2


# In[ ]:


# Path to the saved model weights
weights_path = '/kaggle/input/final-weights-cat/best_model_final_cat.pth'

# Create an instance of your FusionModel
model = FusionModel(fusion_dim, num_classes)

# Load the saved weights
model.load_state_dict(torch.load(weights_path))



# In[ ]:


model


# In[ ]:


#decoding
emotions = ['amusement',
            'awe',
            'contentment',
            'excitement',
            'anger',
            'disgust',
            'fear',
            'sadness',
            'something else' ]


# In[ ]:


decoded_emotions = []
test_set_predictions = []
# Training loop with tqdm
for epoch in range(1):
    # Ensure the model is in evaluation mode
    model.eval()

    model = model.to(device)
    predictions = []

    # Use tqdm to create a progress bar
    dataloader_iterator = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True)

    for i, (image_features, text_features) in dataloader_iterator:

        # Forward pass through the FusionModel
        outputs = model(image_features, text_features).to(device)
        # Move 'output' to the CPU if it's on the GPU (cuda:0)
        if outputs.is_cuda:
            outputs = outputs.cpu()
        max_indices = torch.argmax(outputs, dim=1)

        # Define the number of classes 
        num_classes = 9

        # Convert indices to one-hot representations
        one_hots = torch.zeros((outputs.size(0), num_classes))
        one_hots.scatter_(1, max_indices.view(-1, 1), 1)

        test_set_predictions.append(one_hots)
        # Decode one-hot representations
        for one_hot in one_hots:
        # Find the index of the 1 in the one-hot tensor
            index = torch.argmax(one_hot).item()

        # Get the corresponding emotion label
            emotion_label = emotions[index]
            decoded_emotions.append(emotion_label)


# In[ ]:


import json

def export_json(image_ids, predictions, output_file):
    data = []
    for image_id, prediction in zip(image_ids, predictions):
        data.append({"image_id": int(image_id), "emotion": prediction})
    
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)  # You can use indent to make the JSON file more readable

    print(json.dumps(data, indent=2))


# In[ ]:


out = 'preds_dev.json'

export_json(image_id, decoded_emotions, out)


# In[ ]:


decoded_emotions


# In[ ]:




