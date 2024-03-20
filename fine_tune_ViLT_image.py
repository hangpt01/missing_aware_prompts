import torch
from transformers import ViTFeatureExtractor, ViltProcessor, ViTForImageClassification, CLIPProcessor, ViltModel, AdamW
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pickle 
from torch import nn

# Define your dataset class
class MemeDataset(Dataset):
    def __init__(self, texts, images, labels, processor):
        self.texts = texts
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]
        label = self.labels[idx]

        # Tokenize text and extract image features
        inputs = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        
        # for k, v in inputs.items():
        #     if
        label = torch.tensor(label)
        
        # import pdb; pdb.set_trace()
        
        return inputs, label
    
        # return {
        #     "input_ids": inputs["input_ids"].squeeze(),
        #     "attention_mask": inputs["attention_mask"].squeeze(),
        #     "pixel_values": image_features.squeeze(),
        #     "labels": torch.tensor(label, dtype=torch.float32)
        # }
        
def collate(batch):
    # import pdb; pdb.set_trace()
    batch_ = batch
    batch = []
    labels = []
    for input, label in batch_:
        batch.append(input)
        labels.append(label)
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'][0] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    # import pdb; pdb.set_trace()
    # labels = [item['labels'].item() for item in batch]

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    
    # import pdb; pdb.set_trace()
    batch = {}
    batch['input_ids'] = torch.cat(input_ids)
    batch['attention_mask'] = torch.cat(attention_mask)
    batch['token_type_ids'] = torch.cat(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    # batch['labels'] = torch.tensor(labels)
    return batch, torch.tensor(labels)   

def load_data():
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    return texts, images, labels

# Define your ViLT model (CLIP)
# model_name = "openai/clip-vit-base-patch16"
# tokenizer = CLIPProcessor.from_pretrained(model_name)
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
# model = CLIPModel.from_pretrained(model_name)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
# Freeze the parameters of the ViLT model
for param in model.parameters():
    param.requires_grad = False
# classifier = torch.nn.Linear(768, 2)
classifier = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
)

# Define your dataset and split it into train and validation sets
texts, images, labels = load_data()
# import pdb; pdb.set_trace()

train_texts, val_texts, train_images, val_images, train_labels, val_labels = train_test_split(texts, images, labels, test_size=0.2)

# Initialize datasets and data loaders
train_dataset = MemeDataset(train_texts, train_images, train_labels, processor)
val_dataset = MemeDataset(val_texts, val_images, val_labels, processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate)

# Define training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
classifier.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch, labels in train_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        import pdb; pdb.set_trace()
        # for k,v in batch.items():
        #     print(k, v.shape)
        # optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(**batch)
        features = outputs.last_hidden_state[:, 0, :]
        predictions = classifier(features) 
        loss = criterion(predictions, labels.to(device))
        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch, labels in val_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            features = outputs.last_hidden_state[:, 0, :]
            predictions = classifier(features) 
            val_loss = criterion(predictions, labels.to(device))
            val_losses.append(val_loss.item())

    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_models/small_model")
