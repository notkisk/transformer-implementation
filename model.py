import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import get_config


config = get_config()

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def forward(self, x):
        # first calculate the mean, x is of size(batch_size, sequence_lenth, d_model)
        # layer norm is calculated along the token embedding dimension, so each token will have its own normalized value
        # unlike batch norm, which if used in this case, we would calculate the norm along the sequence, so the mean and standard deviation is calculated over all the tokens
        # meaning that the populaiton here is each token with its embedding, but in layer norm the population the embedding dimetions of each token alone

        mean = x.mean(dim = -1, keepdim = True) # dim = -1 becayse: since we are calculating the layer norm, we do it on the token embedding, which is the last dim from x(batch, seq_len, d_model)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha *  ((x - mean) /(std + self.eps)) + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff, dtype=torch.float32)
        self.linear2 = nn.Linear(d_ff, d_model, dtype=torch.float32)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=torch.float32)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int , seq_len : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #XXX: this implementation is numirically unstable because dividing by a very big number (10000**...) 
        pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
        positions = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        # Compute the division term which is 10000^(2i/d_model)
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        # Apply sin to even indices in the array; 2i
        # So for the div term, it moves like div_term[0:d_model:2] or just div_term[::2]
        
        pe[:,::2] = torch.sin(positions / div_term)
        pe[:,1::2] = torch.cos(positions / div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) # shape (1, seq_len, d_model), we unsqueze because later, this will take input of size (batch_size, sequence_length, d_model), to account for that we add a new dimention to avoid broadcasting error
    def forward(self, x):
        x= x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class ResidualAddNorm(nn.Module):
    def __init__(self,features : int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(features)

    def forward(self, x, sublayer): # x is the input to the sublayer, sublayer is a function like the feedforward or attention.
        # x: (batch_size, seq_len, d_model)
        # Apply residual connection to any sublayer with the same size
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, mha: MultiHeadAttentionBlock, ffn: FeedForward, dropout: float):
        super().__init__()
        self.mha = mha
        self.ffn = ffn
        self.residual_blocks = nn.ModuleList([ResidualAddNorm(mha.d_model, dropout) for _ in range(2)]) # Two residual add norm blocks, one for mha and one for ffn
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask):
        x = self.residual_blocks[0](x, lambda x: self.mha(x,x,x, src_mask)) # Self attention, so q,k,v are all x
        x= self.residual_blocks[1](x, self.ffn)
        return self.dropout(x)
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(config.get('d_model', 512))

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_mha: MultiHeadAttentionBlock, cross_mha: MultiHeadAttentionBlock, ffn: FeedForward, dropout: float):
        super().__init__()
        self.self_mha = self_mha
        self.cross_mha = cross_mha
        self.ffn = ffn
        self.residual_blocks = nn.ModuleList([ResidualAddNorm(self_mha.d_model, dropout) for _ in range(3)]) # Three residual connection blocks, one for self attention, one for cross attention and one for feedforward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.residual_blocks[0](x, lambda x: self.self_mha(x,x,x, trg_mask))
        x = self.residual_blocks[1](x, lambda x: self.cross_mha(x, enc_out, enc_out, src_mask))
        x = self.residual_blocks[2](x, self.ffn)
        return self.dropout(x)
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(config.get('d_model', 512))

    def forward(self, x, enc_out, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return self.layer_norm(x)

class LinearProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, dtype=torch.float32)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, vocab_size)
        return self.log_softmax(self.projection(x))
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: LinearProjection) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = LinearProjection(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer