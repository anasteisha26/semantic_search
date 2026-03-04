import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import random

# =========================
# CONFIG
# =========================

MODEL_NAME = "distilbert-base-uncased"  # только tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 3
LR = 3e-4
HIDDEN_DIM = 256
PROJ_DIM = 256
TEMPERATURE = 0.05


# =========================
# DATASET
# =========================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


# =========================
# MODEL
# =========================

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=300):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.projection = nn.Linear(HIDDEN_DIM * 2, PROJ_DIM)

    def forward(self, input_ids, attention_mask):

        x = self.embedding(input_ids)

        outputs, _ = self.rnn(x)

        # mask-aware mean pooling
        mask = attention_mask.unsqueeze(-1)
        outputs = outputs * mask
        summed = outputs.sum(1)
        counts = mask.sum(1)
        mean_pooled = summed / counts

        mean_pooled = self.dropout(mean_pooled)

        z = self.projection(mean_pooled)
        z = F.normalize(z, dim=-1)

        return z


# =========================
# CONTRASTIVE LOSS (InfoNCE)
# =========================

def contrastive_loss(z1, z2, temperature=TEMPERATURE):
    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)

    similarity_matrix = torch.matmul(representations, representations.T)

    # remove self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(DEVICE)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    similarity_matrix /= temperature

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(DEVICE)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


# =========================
# TRAINING FUNCTION
# =========================

def train_model(texts):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RNNEncoder(tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # two forward passes with dropout (SimCSE trick)
            z1 = model(input_ids, attention_mask)
            z2 = model(input_ids, attention_mask)

            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

    return model, tokenizer


# =========================
# EMBEDDING FUNCTION
# =========================

@torch.no_grad()
def encode(texts, model, tokenizer, batch_size=128):

    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        embeddings = model(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# =========================
# USAGE EXAMPLE
# =========================

if __name__ == "__main__":

    texts = [
        "I love machine learning",
        "Deep learning is fascinating",
        "Cats are beautiful animals",
        "Dogs are loyal pets"
    ]

    model, tokenizer = train_model(texts)

    embeddings = encode(texts, model, tokenizer)

    print("Embeddings shape:", embeddings.shape)
