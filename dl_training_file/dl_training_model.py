import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import re
import numpy as np


# Example tokenizer
def code_tokenizer(code_string):
    return re.findall(r'\w+|[+\-*/=(){};:]', code_string)


# Custom dataset for code pairs
class CodePairDataset(Dataset):
    def __init__(self, csv_file, token_to_index, max_len=50):
        self.data = pd.read_csv(csv_file)
        self.token_to_index = token_to_index
        self.max_len = max_len

    def tokenize_and_pad(self, code):
        tokens = code_tokenizer(code)
        indices = [self.token_to_index.get(token, 0) for token in tokens]  # 0 for unknown tokens
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code1 = row["Code Snippet 1"]
        code2 = row["Code Snippet 2"]
        label = row["Label"]
        tokens1 = self.tokenize_and_pad(code1)
        tokens2 = self.tokenize_and_pad(code2)
        return tokens1, tokens2, torch.tensor(label, dtype=torch.float)


# Model architecture: Siamese Network for semantic duplicate detection
class CodeEmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, token_indices):
        embedded = self.embedding(token_indices)
        _, hidden = self.rnn(embedded)
        return hidden[-1]


class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = CodeEmbeddingRNN(vocab_size, embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        diff = torch.abs(emb1 - emb2)
        out = self.fc(diff)
        return self.sigmoid(out)


# Load token mapping and hyperparameters
with open("token_to_index.pkl", "rb") as f:
    token_to_index = pickle.load(f)
vocab_size = len(token_to_index)
embedding_dim = 50
hidden_dim = 100

# Create datasets and split into training and validation sets
full_dataset = CodePairDataset("rust_semantic_duplicate_data.csv", token_to_index)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, and optimizer
model = SiameseNetwork(vocab_size, embedding_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for tokens1, tokens2, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(tokens1, tokens2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tokens1, tokens2, labels in val_loader:
            outputs = model(tokens1, tokens2).squeeze()
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Optionally, print detailed classification report
    # print(classification_report(all_labels, all_preds, zero_division=0))

torch.save(model.state_dict(), "semantic_duplicate_detector.pth")
