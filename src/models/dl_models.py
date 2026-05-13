import torch
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import log_loss, f1_score, roc_auc_score
import numpy as np

class MultiTaskMLP(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], lr: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Shared representations
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Task specific heads (7 targets: Q1, Q2, Q3, S1, S2, S3, S4)
        self.targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
        
        # Each head outputs a single logit for binary classification
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(hidden_dims[-1], 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for target in self.targets
        })
        
        # Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.validation_step_outputs = []

    def forward(self, x):
        shared_rep = self.shared_layers(x)
        # return a dictionary of logits
        logits = {target: self.heads[target](shared_rep).squeeze(-1) for target in self.targets}
        return logits

    def _compute_loss(self, logits, y, mask):
        loss = 0.0
        active_tasks = 0
        
        for i, target in enumerate(self.targets):
            # y and mask are [batch_size, 7]
            target_logits = logits[target]
            target_y = y[:, i]
            target_mask = mask[:, i]
            
            if target_mask.sum() > 0:
                task_loss = self.criterion(target_logits, target_y)
                # Apply mask to ignore missing targets
                task_loss = (task_loss * target_mask).sum() / target_mask.sum()
                loss += task_loss
                active_tasks += 1
                
        if active_tasks > 0:
            loss = loss / active_tasks
            
        return loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        loss = self._compute_loss(logits, y, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        loss = self._compute_loss(logits, y, mask)
        self.log("val_loss", loss, prog_bar=True)
        
        # Store predictions and labels for epoch-level metrics
        preds = {t: torch.sigmoid(logits[t]).detach().cpu().numpy() for t in self.targets}
        labels = {t: y[:, i].detach().cpu().numpy() for i, t in enumerate(self.targets)}
        masks = {t: mask[:, i].detach().cpu().numpy() for i, t in enumerate(self.targets)}
        
        out = {"val_loss": loss, "preds": preds, "labels": labels, "masks": masks}
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return
            
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        target_losses = []
        
        for t in self.targets:
            t_preds = np.concatenate([x["preds"][t] for x in outputs])
            t_labels = np.concatenate([x["labels"][t] for x in outputs])
            t_masks = np.concatenate([x["masks"][t] for x in outputs])
            
            # Filter by mask
            valid_idx = t_masks > 0
            if valid_idx.sum() > 0:
                valid_preds = t_preds[valid_idx]
                valid_labels = t_labels[valid_idx]
                
                try:
                    t_log_loss = log_loss(valid_labels, valid_preds)
                    target_losses.append(t_log_loss)
                    self.log(f"val_logloss_{t}", t_log_loss)
                except:
                    pass
                    
        if len(target_losses) > 0:
            self.log("val_logloss_mean", np.mean(target_losses), prog_bar=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out, padding_mask=None):
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attention(lstm_out) # (batch_size, seq_len, 1)
        
        if padding_mask is not None:
            # padding_mask is (batch_size, seq_len), True for valid, False for padding
            attn_weights = attn_weights.masked_fill(~padding_mask.unsqueeze(-1), -1e9)
            
        attn_weights = torch.softmax(attn_weights, dim=1)
        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        return context

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return x + self.fc(x)

class MultiTaskLSTM(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-5, num_users: int = 100, user_embed_dim: int = 16):
        super().__init__()
        self.save_hyperparameters()
        
        # 1D CNN Block for local feature extraction
        cnn_out_dim = 64
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        lstm_hidden = hidden_dim // 2
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim, 
            hidden_size=lstm_hidden, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = TemporalAttention(hidden_dim)
        
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # Shared representation after Attention + User Embedding
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim + user_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task specific heads (7 targets: Q1, Q2, Q3, S1, S2, S3, S4)
        self.targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
        
        # Each head outputs a single logit for binary classification
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                ResidualBlock(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for target in self.targets
        })
        
        # Robust loss
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.validation_step_outputs = []

    def forward(self, x, padding_mask=None, user_id=None):
        # x shape: (batch_size, seq_len, input_dim)
        
        # CNN expects (batch_size, channels, seq_len)
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        # Back to (batch_size, seq_len, channels)
        cnn_out = cnn_out.transpose(1, 2)
        
        lstm_out, (hn, cn) = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply temporal attention to weight the sequence days
        context = self.attention(lstm_out, padding_mask)
        
        # Add user embedding
        if user_id is not None:
            # Handle out of bounds just in case
            user_id = torch.clamp(user_id, 0, self.user_embedding.num_embeddings - 1)
            user_emb = self.user_embedding(user_id) # (batch_size, user_embed_dim)
            context = torch.cat([context, user_emb], dim=1)
        else:
            # Fallback to zero embedding if not provided
            dummy_emb = torch.zeros(context.size(0), self.user_embedding.embedding_dim, device=context.device)
            context = torch.cat([context, dummy_emb], dim=1)
            
        shared_rep = self.shared_fc(context)
        
        # return a dictionary of logits
        logits = {target: self.heads[target](shared_rep).squeeze(-1) for target in self.targets}
        return logits

    def _compute_loss(self, logits, y, mask):
        loss = torch.tensor(0.0, device=y.device)
        active_tasks = 0
        
        for i, target in enumerate(self.targets):
            # y and mask are [batch_size, 7]
            target_logits = logits[target]
            target_y = y[:, i]
            target_mask = mask[:, i]
            
            if target_mask.sum() > 0:
                task_loss = self.criterion(target_logits, target_y)
                # Apply mask to ignore missing targets
                task_loss = (task_loss * target_mask).sum() / target_mask.sum()
                loss += task_loss
                active_tasks += 1
                
        if active_tasks > 0:
            loss = loss / active_tasks
        else:
            loss.requires_grad = True # Keep computation graph alive if needed
            
        return loss

    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            x, y, mask, padding_mask, user_id = batch
        elif len(batch) == 4:
            x, y, mask, padding_mask = batch
            user_id = None
        else:
            x, y, mask = batch
            padding_mask = None
            user_id = None
            
        logits = self(x, padding_mask, user_id)
        loss = self._compute_loss(logits, y, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            x, y, mask, padding_mask, user_id = batch
        elif len(batch) == 4:
            x, y, mask, padding_mask = batch
            user_id = None
        else:
            x, y, mask = batch
            padding_mask = None
            user_id = None
            
        logits = self(x, padding_mask, user_id)
        loss = self._compute_loss(logits, y, mask)
        self.log("val_loss", loss, prog_bar=True)
        
        # Store predictions and labels for epoch-level metrics
        preds = {t: torch.sigmoid(logits[t]).detach().cpu().numpy() for t in self.targets}
        labels = {t: y[:, i].detach().cpu().numpy() for i, t in enumerate(self.targets)}
        masks = {t: mask[:, i].detach().cpu().numpy() for i, t in enumerate(self.targets)}
        
        out = {"val_loss": loss, "preds": preds, "labels": labels, "masks": masks}
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            return
            
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        target_losses = []
        
        for t in self.targets:
            t_preds = np.concatenate([x["preds"][t] for x in outputs])
            t_labels = np.concatenate([x["labels"][t] for x in outputs])
            t_masks = np.concatenate([x["masks"][t] for x in outputs])
            
            # Filter by mask
            valid_idx = t_masks > 0
            if valid_idx.sum() > 0:
                valid_preds = t_preds[valid_idx]
                valid_labels = t_labels[valid_idx]
                
                try:
                    t_log_loss = log_loss(valid_labels, valid_preds)
                    target_losses.append(t_log_loss)
                    self.log(f"val_logloss_{t}", t_log_loss)
                except:
                    pass
                    
        if len(target_losses) > 0:
            self.log("val_logloss_mean", np.mean(target_losses), prog_bar=True)
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiTaskTransformer(MultiTaskLSTM):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-5, num_users: int = 100, user_embed_dim: int = 16):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, lr=lr, weight_decay=weight_decay, num_users=num_users, user_embed_dim=user_embed_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.attention = TemporalAttention(hidden_dim)
        
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim + user_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
        
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                ResidualBlock(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for target in self.targets
        })
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.validation_step_outputs = []
        
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def forward(self, x, padding_mask=None, user_id=None):
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        cnn_out = cnn_out.transpose(1, 2)
        
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask
        else:
            src_key_padding_mask = None
            
        cnn_out = cnn_out.transpose(0, 1)
        cnn_out = self.pos_encoder(cnn_out)
        cnn_out = cnn_out.transpose(0, 1)
        
        trans_out = self.transformer_encoder(cnn_out, src_key_padding_mask=src_key_padding_mask)
        
        context = self.attention(trans_out, padding_mask)
        
        if user_id is not None:
            user_id = torch.clamp(user_id, 0, self.user_embedding.num_embeddings - 1)
            user_emb = self.user_embedding(user_id) 
            context = torch.cat([context, user_emb], dim=1)
        else:
            dummy_emb = torch.zeros(context.size(0), self.user_embedding.embedding_dim, device=context.device)
            context = torch.cat([context, dummy_emb], dim=1)
            
        shared_rep = self.shared_fc(context)
        
        logits = {target: self.heads[target](shared_rep).squeeze(-1) for target in self.targets}
        return logits
