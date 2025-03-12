import torch
import torch.nn as nn
import pickle
import re

def code_tokenizer(code_string):
    return re.findall(r'\w+|[+\-*/=(){};:]', code_string)

# Define the encoder architecture (must match the one used during training)
class CodeEmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, token_indices):
        embedded = self.embedding(token_indices)
        _, hidden = self.rnn(embedded)
        return hidden[-1]

# Define the Siamese network architecture for duplicate detection
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

# Load vocabulary and model state
with open("token_to_index.pkl", "rb") as f:
    token_to_index = pickle.load(f)
vocab_size = len(token_to_index)
embedding_dim = 50
hidden_dim = 100

# Instantiate and load the Siamese model
model = SiameseNetwork(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("semantic_duplicate_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Function to get embedding from a code snippet using the encoder part
def get_embedding(code_snippet):
    """Generate an embedding for a code snippet using the encoder from the Siamese network."""
    tokens = code_tokenizer(code_snippet)
    indices = torch.tensor([token_to_index.get(t, 0) for t in tokens])
    with torch.no_grad():
        embedding = model.encoder(indices.unsqueeze(0))
    return embedding.squeeze(0).tolist()

# Function to compute duplicate probability between two code snippets using the full Siamese network
def predict_duplicate(code_snippet1, code_snippet2):
    """Return the probability that the two code snippets are semantic duplicates."""
    tokens1 = code_tokenizer(code_snippet1)
    tokens2 = code_tokenizer(code_snippet2)
    indices1 = torch.tensor([token_to_index.get(t, 0) for t in tokens1])
    indices2 = torch.tensor([token_to_index.get(t, 0) for t in tokens2])
    with torch.no_grad():
        output = model(indices1.unsqueeze(0), indices2.unsqueeze(0))
    return output.item()  # This is the duplicate probability

# Function to compute cosine similarity between two embeddings
def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    emb1 = torch.tensor(embedding1)
    emb2 = torch.tensor(embedding2)
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
