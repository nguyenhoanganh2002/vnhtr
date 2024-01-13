from transformers import MBartForConditionalGeneration, VisionEncoderDecoderModel
import torch.nn as nn
from torch.nn import functional as F
import py_vncorenlp
import sentencepiece
import math
import torch

class VNTrOCR(nn.Module):
    def __init__(self):
        super().__init__(config=None)
        self.ViTEncoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten").encoder
        self.adaptive_layer = nn.Linear(384, 768)
        bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable-base")
        self.BartDecoder = bartpho.model.decoder
        self.fc_out = bartpho.lm_head
        self.pad_token_id = 1

    def forward(self, pixel_values, tgt_inputs, att_mask=None):
        """
        `pixel_values`: B,C,H,W
        `tgt_inputs`: B, tgt_len
        `att_mask`: B, tgt_len
        """
        encoder_output = self.ViTEncoder(pixel_values)[0] # B, src_len, 384
        encoder_output = self.adaptive_layer(encoder_output) # B, src_len 768

        max_input_len = att_mask.sum(dim=-1).max().item()

        decoder_output = self.BartDecoder(input_ids=tgt_inputs[:, :max_input_len-1],
                                    attention_mask=att_mask[:, :max_input_len-1],
                                    encoder_hidden_states=encoder_output) # B, tgt_len, 768
        logits = self.fc_out(decoder_output[0]) # B, tgt_len, vocab_size
        return F.cross_entropy(input=logits.transpose(1,2),
                                        target=tgt_inputs[:, 1:max_input_len],
                                        ignore_index=self.pad_token_id, label_smoothing=0.1)
    
    def forward_logit(self, pixel_values, tgt_inputs, att_mask=None):
        encoder_output = self.ViTEncoder(pixel_values)[0] # B, src_len, 384
        encoder_output = self.adaptive_layer(encoder_output) # B, src_len 768

        max_input_len = att_mask.sum(dim=-1).max().item()

        decoder_output = self.BartDecoder(input_ids=tgt_inputs[:, :max_input_len-1],
                                    attention_mask=att_mask[:, :max_input_len-1],
                                    encoder_hidden_states=encoder_output) # B, tgt_len, 768
        logits = self.fc_out(decoder_output[0]) # B, tgt_len, vocab_size
        return logits, max_input_len
    
    def forward_encode(self, pixel_values):
        encoder_output = self.ViTEncoder(pixel_values)[0] # B, src_len, 384
        encoder_output = self.adaptive_layer(encoder_output) # B, src_len 768
        return encoder_output
    
    def forward_decode(self, prev_ids, encoder_output):
        decoder_output = self.BartDecoder(input_ids=prev_ids,
                                          encoder_hidden_states=encoder_output) # B, tgt_len, 768
        logits = self.fc_out(decoder_output[0]) # B, tgt_len, vocab_size
        return logits

class Rethinking(nn.Module):
    def __init__(self, config, masked=True):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.rank = config.rank
        self.block_size = config.max_seq_len
        self.c_attn = nn.Linear(self.vocab_size, self.rank*3, bias=config.bias)
        self.c_proj = nn.Linear(self.rank, self.vocab_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(0)
        self.resid_dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.masked = masked


    def forward(self, x):
        B, T, C = x.shape                                               #batch_size, block_size, n_embd
        q, k, v = self.c_attn(x).split(self.rank, dim=2)

        # q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)  # (B, n_heads, T, h_size)
        # k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)
        # v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)

        att = q@k.transpose(-2,-1)*(1/math.sqrt(k.size(-1)))            # (B, T, T)
        if self.masked:
            mask = torch.tril(torch.ones((T, T))).view(1, T, T).to(x.device)
            att = att.masked_fill_(mask == 0, -float('inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att@v                                                       #(B, T, n_embd)
        # y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return self.gelu(y)


class AdapterVNTrOCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.basemodel = VNTrOCR()
        self.rethinking = Rethinking(config)
        # self.squeeze = nn.Linear(config.vocab_size, config.rank)
        # self.dropout = nn.Dropout(0.1)
        # self.unsqueeze = nn.Linear(config.rank, config.vocab_size)
    
    def forward(self, pixel_values, tgt_inputs, att_mask=None):
        logits, max_input_len = self.basemodel.forward_logit(pixel_values, tgt_inputs, att_mask)
        logits = logits + self.rethinking(logits)
        # logits = self.dropout(self.squeeze(logits))
        # logits = F.relu(logits)
        # logits = self.unsqueeze(logits)
        return  F.cross_entropy(input=logits.transpose(1,2),
                                target=tgt_inputs[:, 1:max_input_len],
                                ignore_index=self.basemodel.pad_token_id, label_smoothing=0.1)
    
    def forward_adapt(self, logits):
        logits = logits + self.rethinking(logits)
        # logits = self.dropout(self.squeeze(logits))
        # logits = self.unsqueeze(logits)
        return logits

