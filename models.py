# -*- coding: utf-8 -*-
"""models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gW1Yt1BTnAP-VJuRHG8CfTNskjQbzKTI
"""

#!pip install pytorch-lightning

import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F

from sklearn import metrics

class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate=0.1, padding_idx=0):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size, padding_idx=0)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MolTrans(pl.LightningModule):
    pass

class DeepDTA(pl.LightningModule):

    def __init__(self,
                 drug_in_dim,
                 drug_convolution_filter_sizes, 
                 drug_convolution_kernel_sizes,
                 drug_convolution_out_dim,
                 drug_out_dim,
                 drug_sequence_max_length,
                 drug_embedding_dim,
                 protein_in_dim,
                 protein_convolution_filter_sizes, 
                 protein_convolution_kernel_sizes,
                 protein_convolution_out_dim,
                 protein_out_dim,
                 protein_sequence_max_length,
                 protein_embedding_dim,
                 fully_connected_dimensions,
                 num_class,
                 binary=True,
                 verbose_evaluation_metrics=True,
                 evaluation_callbacks=None,
                 evaluation_thresholds=None,
                 ):

        super(DeepDTA, self).__init__()
        
        self.evaluation_thresholds = evaluation_thresholds
        self.evaluation_callbacks = evaluation_callbacks

        drug_convolution_channels = [drug_embedding_dim] + drug_convolution_filter_sizes
        self.drug_encoder_layer_num = len(drug_convolution_filter_sizes)
        self.drug_convolution_kernel_sizes = drug_convolution_kernel_sizes
        self.drug_convolution_out_dim = drug_convolution_out_dim
        self.drug_out_dim = drug_out_dim
        self.drug_embedding_dim = drug_embedding_dim
        self.drug_sequence_max_length = drug_sequence_max_length
        self.drug_encoder = nn.Sequential()
        self.drug_embedding = Embeddings(vocab_size=drug_in_dim, hidden_size=drug_embedding_dim, max_position_size=drug_sequence_max_length, padding_idx=0)


        protein_convolution_channels = [protein_embedding_dim] + protein_convolution_filter_sizes
        self.protein_encoder_layer_num = len(protein_convolution_filter_sizes)
        self.protein_convolution_kernel_sizes = protein_convolution_kernel_sizes
        self.protein_convolution_out_dim = protein_convolution_out_dim
        self.protein_out_dim = protein_out_dim
        self.protein_embedding_dim = protein_embedding_dim
        self.protein_sequence_max_length = protein_sequence_max_length
        self.protein_encoder = nn.Sequential()
        self.protein_embedding = Embeddings(vocab_size=protein_in_dim, hidden_size=protein_embedding_dim, max_position_size=protein_sequence_max_length, padding_idx=0)


        hidden_channels = [drug_out_dim + protein_out_dim] + fully_connected_dimensions
        self.fc_layer_num = len(fully_connected_dimensions)
        self.fc = nn.Sequential()

        self.classifier = nn.Linear(hidden_channels[-1], num_class)

        for idx in range(self.drug_encoder_layer_num):
            self.drug_encoder.add_module(f"conv{idx}", nn.Conv1d(in_channels = drug_convolution_channels[idx],
                                                                 out_channels = drug_convolution_channels[idx + 1],
                                                                 kernel_size = self.drug_convolution_kernel_sizes[idx]))
            self.drug_encoder.add_module("relu", nn.ReLU())
            self.drug_encoder.add_module("dropout", nn.Dropout(p=0.1))
            
        self.drug_encoder.add_module("adaptivemaxpool1d", nn.AdaptiveAvgPool1d(output_size=1))
        self.drug_encoder.add_module("flatten", nn.Flatten())
        self.drug_encoder.add_module("fc", nn.Linear(self.drug_convolution_out_dim, self.drug_out_dim))
        self.drug_encoder.add_module("batch_norm_fc", nn.BatchNorm1d(self.drug_out_dim))
        
        
        for idx in range(self.protein_encoder_layer_num):
            self.protein_encoder.add_module(f"conv{idx}", nn.Conv1d(in_channels = protein_convolution_channels[idx],
                                                                    out_channels = protein_convolution_channels[idx + 1],
                                                                    kernel_size = self.protein_convolution_kernel_sizes[idx]))
            self.protein_encoder.add_module("relu", nn.ReLU())
            self.protein_encoder.add_module("dropout", nn.Dropout(p=0.1))
            
        self.protein_encoder.add_module("adaptivemaxpool1d", nn.AdaptiveAvgPool1d(output_size=1))
        self.protein_encoder.add_module("flatten", nn.Flatten())
        self.protein_encoder.add_module("fc", nn.Linear(self.protein_convolution_out_dim, self.protein_out_dim))

        for idx in range(self.protein_encoder_layer_num):
            self.fc.add_module(f"fc{idx +1}", nn.Linear(hidden_channels[idx],
                                                        hidden_channels[idx + 1]))
            self.fc.add_module("relu", nn.ReLU())
            self.fc.add_module("dropout", nn.Dropout(p=0.1))

        self.binary = binary
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        if not self.binary:
          self.loss_fn =  torch.nn.MSELoss()
          

    def forward(self, drug, protein):

        drug = self.drug_encoder(self.drug_embedding(drug).view(-1, self.drug_embedding_dim, self.drug_sequence_max_length))
        protein = self.protein_encoder(self.protein_embedding(protein).view(-1, self.protein_embedding_dim, self.protein_sequence_max_length))

        out = torch.cat([drug, 
                         protein],
                        dim=1)

        out = self.fc(out)
        out = self.classifier(out)

        out = torch.squeeze(out, 1)

        return out

    def training_step(self, batch, batch_idx):

        drug, protein, target = batch
        target = target.float()

        y_hat = self(drug, protein)
        loss = self.loss_fn(y_hat, target)

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {
            'loss': loss,
            "preds": y_hat.detach().cpu(),
            "target": target.detach().cpu()
        }


    def validation_step(self, batch, batch_idx):
        
        drug, protein, target = batch
        target = target.float()

        y_hat = self(drug, protein)
        loss = self.loss_fn(y_hat, target)

        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {
            'loss': loss,
            "preds": y_hat.detach().cpu(),
            "target": target.detach().cpu(),
        }
  
    def validation_epoch_end(self, outputs):
        
        metric = "avg_val"
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])

        print(f"Epoch {self.current_epoch}\n")

        res = {
            f'{metric}_loss': avg_loss,
            'log': {f'{metric}_loss': avg_loss},
        }

        for eval_callback in self.evaluation_callbacks:
            for threshold in self.evaluation_thresholds:

              if eval_callback in ["f1_score", "recall_score", "precision_score"]:
                  eval_score = getattr(metrics, eval_callback)(target, (preds > threshold) * 1)

              else:
                  eval_score = getattr(metrics, eval_callback)(target, preds)

              self.log(f"{eval_callback}",
                        eval_score,
                        logger=True)
              
              print(f"{eval_callback} -- {eval_score}")

        print("\n")

        return res

    def test_step(self, batch, batch_idx):

        drug, protein = batch
        y_hat = self(drug, protein)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=10)


        return [optimizer], [scheduler]

class DeepConvDTI(pl.LightningModule):

    def __init__(self,
                 protein_in_dim,
                 protein_embedding_dim,
                 protein_max_seq_length,
                 protein_filter_size,
                 protein_kernel_sizes,
                 protein_dense_out_dim,
                 drug_max_seq_length,
                 drug_hidden_dims,
                 fc_out_dim,
                 num_class,
                 binary=True,
                 evaluation_callbacks=None,
                 evaluation_thresholds=None,
                 ):
    
        super(DeepConvDTI, self).__init__()

        self.evaluation_thresholds = evaluation_thresholds
        self.evaluation_callbacks = evaluation_callbacks

        self.protein_embedding_dim = protein_embedding_dim
        self.protein_max_seq_length = protein_max_seq_length
        self.protein_filter_size = protein_filter_size
        self.protein_kernel_sizes = protein_kernel_sizes
        self.protein_dense_in_dim = protein_filter_size * len(protein_kernel_sizes)
        self.protein_dense_out_dim = protein_dense_out_dim

        self.drug_out_dim = drug_hidden_dims[-1]

        self.fc_out_dim = fc_out_dim
        self.num_class = num_class

        self.drug = nn.Sequential()
        drug_channels = [drug_max_seq_length] + drug_hidden_dims

        self.protein_embedding = nn.Embedding(num_embeddings=26, embedding_dim=protein_embedding_dim, padding_idx=0)

        self.conv_filters = nn.ModuleDict({f"conv{idx}": self.ConvBlock(self.protein_embedding_dim, 
                                                                   self.protein_filter_size, 
                                                                   kernel_size)
                                   for idx, kernel_size in enumerate(self.protein_kernel_sizes)})

        self.protein_fc = self.DenseBlock(self.protein_filter_size * len(self.protein_kernel_sizes), self.protein_dense_out_dim)

        self.binary = binary
        self.loss_fn = torch.nn.BCELoss()
        self.last_activation_fn = nn.Sigmoid()

        if not self.binary:
          self.loss_fn =  torch.nn.MSELoss()


        for idx in range(2):
            self.drug.add_module(f"drug{idx}", self.DenseBlock(drug_channels[idx], drug_channels[idx + 1]))

        self.fc = nn.Sequential(nn.Linear(self.drug_out_dim + self.protein_dense_out_dim, self.fc_out_dim),
                    nn.BatchNorm1d(self.fc_out_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_out_dim, self.num_class))
        
    def forward(self, drug, protein):
        
        protein = self.protein_embedding(protein).view(-1, self.protein_embedding_dim, self.protein_max_seq_length)
        protein = torch.cat([conv_filter(protein) for conv_filter in self.conv_filters.values()], dim=1)
        protein = self.protein_fc(protein)
        
        out = torch.cat([protein, self.drug(drug)], axis=1)
        out = self.fc(out)

        if self.binary:
          out = self.last_activation_fn(out)

        out = torch.squeeze(out, 1)

        return out

    def training_step(self, batch, batch_idx):

        drug, protein, target = batch
        target = target.float()

        y_hat = self(drug, protein)
        loss = self.loss_fn(y_hat, target)
  
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {
            'loss': loss,
            "preds": y_hat.detach().cpu(),
            "target": target.detach().cpu()
        }

    def validation_step(self, batch, batch_idx):
        
        drug, protein, target = batch
        target = target.float()

        y_hat = self(drug, protein)
        loss = self.loss_fn(y_hat, target)

        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {
            'loss': loss,
            "preds": y_hat.detach().cpu(),
            "target": target.detach().cpu()
        }
  
    def validation_epoch_end(self, outputs):
        
        metric = "avg_val"
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])

        res = {
            f'{metric}_loss': avg_loss,
            'log': {f'{metric}_loss': avg_loss},
        }

        for eval_callback in self.evaluation_callbacks:
            for threshold in self.evaluation_thresholds:

              if eval_callback in ["f1_score", "recall_score", "precision_score"]:
                  eval_score = getattr(metrics, eval_callback)(target, (preds > threshold) * 1)

              else:
                  eval_score = getattr(metrics, eval_callback)(target, preds)

              self.log(f"{eval_callback}",
                       eval_score,
                       logger=True)
              
              print(f"{eval_callback} -- {eval_score}")

        print("\n")

        return res

    def test_step(self, batch, batch_idx):

        drug, protein = batch
        y_hat = self(drug, protein)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=10)


        return [optimizer], [scheduler]

    def ConvBlock(self, in_channel, filter_size, kernel_size):

        return nn.Sequential(nn.Conv1d(in_channels=in_channel,
                                      out_channels=filter_size,
                                      kernel_size=kernel_size),
                            nn.BatchNorm1d(filter_size),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool1d(output_size=1),
                            nn.Flatten())

    def DenseBlock(self, in_channel, out_channel, p=0.2):
        
        return nn.Sequential(nn.Linear(in_channel, out_channel),
                            nn.BatchNorm1d(out_channel),
                            nn.ReLU(),
                            nn.Dropout(p=p))

