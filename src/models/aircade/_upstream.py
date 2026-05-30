"""Verbatim AirCade model source.

Vendored from https://github.com/PoorOtterBob/AirCade (src/models/AirCade.py).
Only the ``BaseModel`` import path was changed to the shared in-tree base;
the model code below is otherwise unmodified. The benchmark-facing adapter
lives in ``models.aircade.model``.

Vendored under the upstream project's original license; see THIRD_PARTY_NOTICES.md at the repository root.
"""

import torch
import torch.nn as nn
from models._external.base import BaseModel
###AirCade###
class AirCade(BaseModel):

    def __init__(self, time_step = 24, input_embedding_dim = 16, DK_Prompt_adaptive_embedding = 24, feed_forward_dim = 160, num_heads = 8, num_layers = 3,
        dropout = 0.1, DK_Prompt_position_embedding_dim = 16, output_mixed = True, num_nodes = 184, **args):
        super(AirCade, self).__init__(**args)
        self.time_step = time_step
        self.num_nodes = num_nodes
        self.DPpe = DK_Prompt_position_embedding_dim 
        self.DPae = DK_Prompt_adaptive_embedding
        self.input_embedding_dim = input_embedding_dim
        self.model_dim = input_embedding_dim + DK_Prompt_adaptive_embedding
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_mixed = output_mixed
        self.input_proj_x = nn.Linear(self.input_dim, input_embedding_dim) 
        self.input_proj_y = nn.Linear(self.input_dim - 1, input_embedding_dim) 

        ### DK-Prompt ###
        if self.DPae > 0:
            self.adaptive_embedding_x = nn.Parameter(torch.empty(time_step, self.num_nodes, self.DPae))
            nn.init.xavier_uniform_(self.adaptive_embedding_x)
            self.adaptive_embedding_y = nn.Parameter(torch.empty(time_step, self.num_nodes, self.DPae))
            nn.init.xavier_uniform_(self.adaptive_embedding_y)
        if self.DPpe > 0:
            self.position_embedding_x = nn.Parameter(torch.empty(time_step, self.DPpe))
            nn.init.xavier_uniform_(self.position_embedding_x)
            self.position_embedding_y = nn.Parameter(torch.empty(time_step, self.DPpe))
            nn.init.xavier_uniform_(self.position_embedding_y)
        ### DK-Prompt ###

        self.Gated_transformer_encode = nn.ModuleList(
            [   
                Attention_Mechanism(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.Inverse_transformer_encode = nn.ModuleList(
            [
                Attention_Mechanism(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.Gated_transformer_decode = nn.ModuleList(
            [
                Attention_Mechanism(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.Inverse_transformer_deocde = nn.ModuleList(
            [
               Attention_Mechanism(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        if output_mixed:  
            self.output_proj = nn.Linear(time_step * self.model_dim, time_step * self.output_dim)
        else:
            self.temporal_proj = nn.Linear(time_step,time_step)  
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)  

    def forward(self, x, y):
        batch_size = x.shape[0]
        if self.DPpe > 0: 
            position_emb_x = self.position_embedding_x.expand(batch_size, self.num_nodes, *self.position_embedding_x.shape).transpose(1,2)
            position_emb_y = self.position_embedding_y.expand(batch_size, self.num_nodes, *self.position_embedding_y.shape).transpose(1,2)
        x = self.input_proj_x(x.float()) + position_emb_x
        y = self.input_proj_y(y.float()) + position_emb_y
        features_x = [x]
        features_y = [y]
        if self.DPae > 0:
            adp_emb_x = self.adaptive_embedding_x.expand(size=(batch_size, *self.adaptive_embedding_x.shape))
            adp_emb_y = self.adaptive_embedding_y.expand(size=(batch_size, *self.adaptive_embedding_y.shape))
        features_x.append(adp_emb_x)
        features_y.append(adp_emb_y)
        x = torch.cat(features_x, dim = -1) 
        y = torch.cat(features_y, dim = -1)

        for Cade_t, ES_t in zip(self.Gated_transformer_encode, self.Gated_transformer_decode):
            x = Cade_t(x, y, True, dim = 1)
            y = ES_t(x, y, False, dim = 1)
        for Cade_s, ES_s in zip(self.Inverse_transformer_encode, self.Inverse_transformer_deocde):
            x = Cade_s(x, y, True, dim = 2)
            y = ES_s(x, y, False, dim = 2)

        x = y

        if self.output_mixed:
            x = x.transpose(1, 2)    
            x = x.reshape(batch_size, x.shape[1], self.time_step * self.model_dim)
            x = self.output_proj(x).view(batch_size, x.shape[1], self.time_step , self.output_dim) 
            x = x.transpose(1, 2)  
        else:
            x = x.transpose(1, 3)  
            x = self.temporal_proj(x) 
            x = self.output_proj(x.transpose(1, 3))  
        return x 
    
class DK_MSA(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim  
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        c_in = 2 * self.head_dim * 2
        self.mlp = nn.Linear(c_in,self.head_dim) 
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.node_emb1 = nn.Parameter(torch.randn(184, 10), requires_grad=True)
        self.node_emb2 = nn.Parameter(torch.randn(10,184), requires_grad=True)

    def forward(self, query, key, value,dim):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0) 
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(
            -1, -2
            )  
        attn_score = (
                query @ key    
            ) / self.head_dim**0.5  
        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  
            attn_score.masked_fill_(~mask, -torch.inf) 
        out_init =[]
        attn_score_row = torch.softmax(attn_score, dim=-1)
        attn_score_col = torch.softmax(attn_score, dim=-2)  
        out_init.append(attn_score_row @ value)
        out_init.append((attn_score_col).transpose(-1,-2) @ value) 
        adp_row = torch.softmax(self.node_emb1 @ self.node_emb2, dim=-1)
        adp_col = torch.softmax(self.node_emb1 @ self.node_emb2, dim=0)
        if dim == 2:
            out_init.append(torch.einsum('ik, btkf -> btif', adp_row, value))
            out_init.append(torch.einsum('ik, btkf -> btif', adp_col.transpose(-1,-2), value))
            out = torch.cat(out_init, dim = -1)
            out = self.mlp(out)
        if dim == 1:
            out_init.append(torch.einsum('ik, bktf -> bitf', adp_row, value))
            out_init.append(torch.einsum('ik, bktf -> bitf', adp_col.transpose(-1,-2), value))
            out = torch.cat(out_init, dim = -1)
            filter = self.mlp(out)
            out_former = torch.tanh(out[:,:,:, : self.head_dim ])
            out_behind = torch.sigmoid(out[:,:,:,-self.head_dim : ])
            gate = out_former * out_behind 
            out = filter + gate 
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  
        out = self.out_proj(out)
        return out
    
class Attention_Mechanism(nn.Module):
    def __init__(self, model_dim, feed_forward_dim = 2048, num_heads = 8, dropout=0, mask=False
    ):
        super().__init__()
        self.Cade = DK_MSA(model_dim, num_heads, mask)
        self.ES = DK_MSA(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim,feed_forward_dim), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.15),
            nn.Linear(feed_forward_dim, model_dim),
            )
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x , y , flag , dim = 1):  
        x = x.transpose(dim, 2) 
        y = y.transpose(dim, 2)
        residual = x
        if flag:
            out = self.Cade(x, x, x, dim)  
        else:
            residual = y
            out = self.ES(y, y, x, dim)   
        out = self.dropout1(out)
        out = self.ln1(residual + out) 
        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out) 
        out = out.transpose(dim, -2) 
        return out

