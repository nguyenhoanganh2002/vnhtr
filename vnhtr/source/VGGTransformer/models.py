import torch.nn as nn
import torch
from config import predictor, device
from submodules import *

class AdapterVGGTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.basemodel = VGGTransformer(config)

        self.rethinking = Rethinking(config)
        self.fc_out = nn.Linear(config.vocab_size, config.vocab_size)
    
    def forward(self, img, tgt, tgt_key_padding_mask):
        en_out = self.basemodel.backbone(img)
        de_out = self.basemodel.transformerLM(en_out, tgt.transpose(1,0), tgt_key_padding_mask=tgt_key_padding_mask)
        # print("------------here 4------------")
        de_out = de_out + self.rethinking(de_out)
        return self.fc_out(de_out)
    
    def forward_adapt(self, de_out):
        de_out = de_out + self.rethinking(de_out)
        return self.fc_out(de_out)
        # tgt_mask = self.basemodel.transformerLM.gen_nopeek_mask(tgt.shape[0]).to(img.device)


class VGGTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.backbone = predictor.cnn.float()

        self.transformerLM = TransformerLM(config)
    
    def forward(self, img, tgt, tgt_key_padding_mask):
        en_out = self.backbone(img)
        return self.transformerLM(en_out, tgt.transpose(1,0), tgt_key_padding_mask=tgt_key_padding_mask)


class ResnetTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.backbone = Resnet34(config)

        self.transformerLM = TransformerLM(config)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, img, tgt, tgt_key_padding_mask):
        en_out = self.backbone(img)
        return self.transformerLM(en_out.transpose(1,0), tgt.transpose(1,0), tgt_key_padding_mask=tgt_key_padding_mask)

class GCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.backbone = ConvEmbeddingGC()

        self.transformerLM = TransformerLM(config)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, img, tgt, tgt_key_padding_mask):
        en_out = self.backbone(img)
        return self.transformerLM(en_out.transpose(1,0), tgt.transpose(1,0), tgt_key_padding_mask=tgt_key_padding_mask)


class FCNTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len

        self.backbone = FCN_Encoder({"dropout": 0.1, "input_channels": 3})

        self.backbone.load_state_dict(torch.load("source/weights/iam_line.pt", map_location=device)["encoder_state_dict"])

        self.transformerLM = TransformerLM(config)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, img, tgt, tgt_key_padding_mask):
        en_out = self.backbone(img)
        return self.transformerLM(en_out.transpose(1,0), tgt.transpose(1,0), tgt_key_padding_mask=tgt_key_padding_mask)