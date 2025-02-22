# inference.py
import torch
import torch.nn as nn
import pickle
import re

class CodeEmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, token_indices):
        embedded = self.embedding(token_indices)
        _, hidden = self.rnn(embedded)
        return hidden[-1]

def code_tokenizer(code_string):
    return re.findall(r'\w+|[+\-*/=(){};:]', code_string)

# Load vocabulary and model
with open("token_to_index.pkl", "rb") as f:
    token_to_index = pickle.load(f)
vocab_size = len(token_to_index)

embedding_dim = 50
hidden_dim = 100
model = CodeEmbeddingRNN(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("semantic_duplicate_detector.pth"))
model.eval()

def get_embedding(code_snippet):
    """Generate embedding for a code snippet."""
    tokens = code_tokenizer(code_snippet)
    indices = torch.tensor([token_to_index.get(t, 0) for t in tokens])
    with torch.no_grad():
        embedding = model(indices.unsqueeze(0))
    return embedding.squeeze(0).tolist()  # Convert to list for Rust

def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    emb1 = torch.tensor(embedding1)
    emb2 = torch.tensor(embedding2)
    return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()