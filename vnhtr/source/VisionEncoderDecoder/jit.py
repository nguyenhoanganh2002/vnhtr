from model import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from evaluate import load
from config import config
import torch.nn as nn
from torch.nn import functional as F
import torch

class TrOCR():
    def __init__(self, device="cuda:2"):
        self.encoder = torch.jit.load("vnhtr/weights/tr_encoder.pt", map_location=device)
        self.decoder = torch.jit.load("vnhtr/weights/tr_decoder.pt", map_location=device)
        self.encoder.eval()
        self.decoder.eval()
        self.device = device
    
    def predict(self, pixel_values, max_seq_length=35, sos_token=0, eos_token=2):
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                encoder_output = self.encoder(pixel_values.to(self.device))
    
                start_ids = torch.LongTensor([[sos_token]]*pixel_values.shape[0]).to(self.device)

                max_length = 0

                while max_length <= max_seq_length and not all(start_ids[:, -1] == eos_token):
                    output = self.decoder(start_ids, encoder_output, self.gen_nopeek_mask(start_ids.shape[1]).to(self.device))
                    output = F.softmax(output[:,-1,:], dim=-1)

                    start_ids = torch.cat([start_ids, output.argmax(dim=-1).unsqueeze(1)], dim=-1)

                    max_length += 1
            
            return start_ids
    
    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ViT(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vit = model.ViTEncoder
        self.adaptive_layer = model.adaptive_layer

    def forward(self, pixel_values):
        encoder_output = self.vit(pixel_values)[0] # B, src_len, 384
        encoder_output = self.adaptive_layer(encoder_output) # B, src_len 768
        return encoder_output

class TrocrDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.BartDecoder = model.BartDecoder
        self.fc = model.fc_out
        # self.rethinking = model.rethinking

    def forward(self, prev_ids, encoder_output, mask):
        decoder_output = self.BartDecoder(input_ids=prev_ids,
                                          encoder_hidden_states=encoder_output) # B, tgt_len, 768
        logits = self.fc(decoder_output[0]) # B, tgt_len, vocab_size
        # logits = logits + self.rethinking(logits, mask)
        return logits
    
def gen_nopeek_mask(length):
    mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

device = 'cuda:3'

if __name__ == "__main__":
    model = VNTrOCR()
    model.load_state_dict(torch.load("/mnt/disk4/VN_HTR/VN_HTR/VisionEncoderDecoder/weights/cp_finetune_v2_5.pt", map_location=device))
    model.to(device)

    anot = pd.read_csv("test_anot.csv").sample(frac=1, random_state=0).reset_index(drop=True)

    valdataset = CustomDataset(anot, config)
    valloader = DataLoader(dataset=valdataset, batch_size=2, shuffle=False, generator=torch.Generator(device="cpu"))

    for batch in valloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        encoder = ViT(model)
        encoder.eval()
        trace_encoder = torch.jit.trace(encoder, pixel_values)
        trace_encoder.save("/mnt/disk4/VN_HTR/VN_HTR/vnhtr/weights/tr_encoder.pt")
        memory = encoder(pixel_values)#.transpose(1, 0)
        # print(memory.shape)
        # print(in_tokens[:, :5].shape)

        decoder = TrocrDecoder(model)
        decoder.eval()
        trace_decoder = torch.jit.trace(decoder, (input_ids[:,:2], memory, gen_nopeek_mask(input_ids[:,:2].shape[1]).to(device)))
        trace_decoder.save("/mnt/disk4/VN_HTR/VN_HTR/vnhtr/weights/tr_decoder.pt")
        
        break