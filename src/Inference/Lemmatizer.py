import torch.nn as nn
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from typing import Tuple, List

class WordEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128, out_dim=128):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=33)
        self.lstm       = nn.LSTM(input_size=emb_dim, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.fc         = nn.Linear(in_features=hidden*2, out_features=out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x.shape = batch x 20 (batched vectors 20 x 1)
        emb = self.embeddings(x) # emb.shape = batch x 20 x 64 (each vec element(token) -> embedding vec)

        _, (h, c) = self.lstm(emb) # h.shape = 2 x batch x 128

        h = torch.cat((h[0], h[1]), dim=1) # forward and backward h concat | h.shape = batch x 256

        return self.fc(h) # batch x 128


class Lemmatizer(nn.Module):
    def __init__(self, char_vocab_size: int, pos_classes: int = 16, max_delete:int = 6, suffix_vocab:int = 512):
        super().__init__()
        self.word2vec = WordEncoder(char_vocab_size, emb_dim=64, hidden=128, out_dim=128)

        self.pos_head    = nn.Linear(in_features=128*5, out_features=pos_classes)
        self.delete_head = nn.Linear(in_features=128*5, out_features=max_delete)
        self.suffix_head = nn.Linear(in_features=128*5, out_features=suffix_vocab)

    def forward(self, target: torch.Tensor, ctx: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_emb = self.word2vec(target)
        ctx_emb    = [self.word2vec(c) for c in ctx]
        
        features_vec  = torch.cat([target_emb] + ctx_emb, dim=1)
        pos_logits    = self.pos_head(features_vec)
        delete_logits = self.delete_head(features_vec)
        suf_logits    = self.suffix_head(features_vec)

        return pos_logits, delete_logits, suf_logits
    
# BCE
def train(n_epoch: int, model: Lemmatizer, train_data: DataLoader, test_data: DataLoader) -> Lemmatizer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = RMSprop(
        model.parameters(),
        lr=0.001,
        alpha=0.99,
        eps=1e-8,
        weight_decay=1e-4,
        momentum=0.0
    )
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.0
        total = 0
        for x, y in train_data:
            target, context = x
            pos, delete, suf = y
            
            target = target.to(device)
            context = [c.to(device) for c in context]
            pos = pos.to(device)
            delete = delete.to(device)
            suf = suf.to(device)

            output_pos, output_delete, output_suf = model(target, context)

            loss = criterion(output_pos, pos) + criterion(output_delete, delete) + criterion(output_suf, suf)
            train_loss += loss.cpu().item()
            total += len(target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= total
        val_loss = 0.0
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_data:
                target, context = x
                pos, delete, suf = y
                
                target = target.to(device)
                context = [c.to(device) for c in context]
                pos = pos.to(device)
                delete = delete.to(device)
                suf = suf.to(device)

                output_pos, output_delete, output_suf = model(target, context)

                loss = criterion(output_pos, pos) + criterion(output_delete, delete) + criterion(output_suf, suf)
                val_loss += loss.cpu().item()
                total += len(target)
            val_loss /= total

        print(f"Epoch {epoch + 1}/{n_epoch} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")
    return model
            
