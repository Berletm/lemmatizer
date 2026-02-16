import torch.nn as nn
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, List
from tqdm import tqdm
import numpy as np

from Data.Tokenizer import Tokenizer, PADDING_TOKEN, MAX_WORD_LEN

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
    def __init__(self, char_vocab_size: int, pos_classes: int = 16, max_delete: int = 6, suffix_vocab: int = 512, context_window:int = 2):
        super().__init__()
        self.word2vec = WordEncoder(char_vocab_size, emb_dim=128, hidden=128, out_dim=128)

        self.pos_head    = nn.Linear(in_features=128*(2*context_window + 1), out_features=pos_classes)
        self.delete_head = nn.Linear(in_features=128*(2*context_window + 1), out_features=max_delete)
        self.suffix_head = nn.Linear(in_features=128*(2*context_window + 1), out_features=suffix_vocab)

        self.context_size = context_window

    def forward(self, target: torch.Tensor, ctx: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_emb = self.word2vec(target)
        ctx_emb    = [self.word2vec(c) for c in ctx]
        
        features_vec  = torch.cat([target_emb] + ctx_emb, dim=1)
        pos_logits    = self.pos_head(features_vec)
        delete_logits = self.delete_head(features_vec)
        suf_logits    = self.suffix_head(features_vec)

        return pos_logits, delete_logits, suf_logits
    
    def lemmatize(self, sentence: str, tokenizer: Tokenizer) -> str:
        tokens = self.__preprocess(sentence, tokenizer)

        data   = self.__prepare_context(tokens, tokenizer, 2)

        temp = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)
        for target, ctx in data:
            target_str = target
            target = torch.tensor(self.__pad(target, tokenizer), dtype=torch.long).to(device).unsqueeze(0)
            ctx    = [torch.tensor(self.__pad(c, tokenizer), dtype=torch.long).to(device).unsqueeze(0) for c in ctx]

            pos_logits, del_logits, suf_logits = self.forward(target, ctx)

            pos    = torch.argmax(pos_logits, dim=1).cpu().item()
            delete = torch.argmax(del_logits, dim=1).cpu().item()
            suf    = torch.argmax(suf_logits, dim=1).cpu().item()

            pos = tokenizer.decode_pos(int(pos))
            suf = tokenizer.decode_suf(int(suf))

            suf = suf if suf else ""

            target_str = tokenizer.detokenize(target_str)
            target_str_cut = target_str[:-delete] if delete != 0 else target_str
            temp.append(f"{target_str}({target_str_cut + suf}={pos})")
        return " ".join(temp)


    
    def __preprocess(self, sentence: str, tokenizer: Tokenizer) -> List[np.ndarray]:
        PUNCT = ",.:;"
        removed_punct = sentence.lower().replace("ё", "е")
        
        for p in PUNCT:
            removed_punct = removed_punct.replace(p, "")
        
        words = removed_punct.split(" ")

        tokenized_words = [tokenizer.tokenize(w) for w in words]

        return tokenized_words

    def __prepare_context(self, tokens: List[np.ndarray], tokenizer: Tokenizer, context_size: int) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        res = []

        for i, tok in enumerate(tokens):
            cur_context = []
            for j in range(i - context_size, i + context_size + 1):
                if j == i:
                    continue
                if j < 0 or j >= len(tokens):
                    cur_context.append(tokenizer(PADDING_TOKEN))
                else:
                    cur_context.append(tokens[j])
            res.append((tok, cur_context))
        
        return res

    def __pad(self, vec: np.ndarray, tokenizer: Tokenizer) -> np.ndarray:
        temp = [int(x) for x in vec]
        pad_value = tokenizer.str2tok[PADDING_TOKEN]
        pad_len = MAX_WORD_LEN - len(temp)

        if pad_len <= 0:
            return vec[:MAX_WORD_LEN].astype(np.int64)
        
        padded = temp + [pad_value] * pad_len
        return np.array(padded, dtype=np.int64)

# BCE
def train(n_epoch: int, model: Lemmatizer, train_data: DataLoader, test_data: DataLoader) -> Lemmatizer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)
    model = model.to(device)

    patience   = 20
    counter    = 0
    best_loss  = float("inf") 
    best_model = model

    for epoch in range(n_epoch):
        if counter >= patience:
            print(f"Early stopping {epoch + 1}/{n_epoch}")
            return best_model
        model.train()
        train_loss = 0.0
        total = 0

        train_loader = tqdm(train_data, desc=f"Epoch {epoch+1}/{n_epoch} [train]", leave=True)

        for x, y in train_loader:
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
            scheduler.step()

            train_loader.set_postfix(loss=f"{train_loss/total:.4f}")

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

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            counter = 0
        else: counter += 1
        print(f"Epoch {epoch + 1}/{n_epoch} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")
    return model
            
def test_lemmatizer(model: Lemmatizer, test_loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total = 0
    correct_pos = 0
    correct_del = 0
    correct_suf = 0
    val_loss = 0.0
    for x, y in test_loader:
        target, context = x
        pos, delete, suf = y
        
        target = target.to(device)
        context = [c.to(device) for c in context]
        pos = pos.to(device)
        delete = delete.to(device)
        suf = suf.to(device)

        output_pos, output_delete, output_suf = model(target, context)

        pred_pos    = torch.argmax(output_pos, dim=1)
        pred_delete = torch.argmax(output_delete, dim=1)
        pred_suf    = torch.argmax(output_suf, dim=1)

        correct_pos += (pred_pos == pos).sum()
        correct_del += (pred_delete == delete).sum()
        correct_suf += (pred_suf == suf).sum()
        total += target.size(0)

        loss = criterion(output_pos, pos) + criterion(output_delete, delete) + criterion(output_suf, suf)
        val_loss += loss.cpu().item()
    print(f"Val loss {val_loss/total:.4f} | Pos acc {correct_pos/total:.4f} | Del acc {correct_del/total:.4f} | Suf acc {correct_suf/total:.4f}")