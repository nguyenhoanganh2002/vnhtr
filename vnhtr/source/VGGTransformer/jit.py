import torch.nn as nn
import torch
# from config import predictor, device
# from submodules import *
# import math
# from models import *
# from utils import *
# from dataset import *
# from torch.utils.data import DataLoader
# import pandas as pd
import numpy as np
from torch.nn import functional as F

class OCRModel():
    def __init__(self, adapter=False, device="cuda:2"):
        if adapter:
            self.encoder = torch.jit.load("vnhtr/weights/vta_encoder.pt", map_location=device)#.to(device)
            self.decoder = torch.jit.load("vnhtr/weights/vta_decoder.pt", map_location=device)#.to(device)
        else:
            self.encoder = torch.jit.load("vnhtr/weights/vt_encoder.pt", map_location=device)#.to(device)
            self.decoder = torch.jit.load("vnhtr/weights/vt_decoder.pt", map_location=device)#.to(device)
        print("loaded model")
        self.encoder.eval()
        self.decoder.eval()
        self.device = device
    
    def predict(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                mem = self.encoder(img.to(self.device))
                translated_sentence = [[sos_token]*len(img)]
                max_length = 0

                while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

                    tgt_inp = torch.LongTensor(translated_sentence).to(self.device)
                    # print(self.gen_nopeek_mask(tgt_inp.shape[0]).to(self.device).device)
                    output = self.decoder(tgt_inp, mem, self.gen_nopeek_mask(tgt_inp.shape[0]).to(self.device))
                    output = F.softmax(output, dim=-1)
                    output = output.to('cpu')

                    values, indices  = torch.topk(output, 5)
                    
                    indices = indices[:, -1, 0]
                    indices = indices.tolist()

                    translated_sentence.append(indices)   
                    max_length += 1

                    del output

                translated_sentence = np.asarray(translated_sentence).T
            
                return translated_sentence
    
    def predict_topk(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        conflict=[]
        with torch.no_grad():
            mem = self.encoder(img.to(self.device))
            translated_sentence = [[sos_token]*len(img)]

            probs = []
            tokens = []

            max_length = 0

            while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

                tgt_inp = torch.LongTensor(translated_sentence).to(self.device)
                
                output = self.decoder(tgt_inp, mem, self.gen_nopeek_mask(tgt_inp.shape[0]).to(self.device))
                output = F.softmax(output, dim=-1)
                output = output.to('cpu')

                values, indices  = torch.topk(output, 233)

                probs.append(values[0, -1].cpu().numpy())
                tokens.append(indices[0, -1].cpu().numpy())
            
                indices = indices[:, -1, 0]
                indices = indices.tolist()

                translated_sentence.append(indices)   
                max_length += 1

                conflict.append(output[:,-1,:].cpu().numpy())

                del output

            translated_sentence = np.asarray(translated_sentence).T
        return translated_sentence, np.array(tokens).T, np.array(probs).T, np.array(conflict)
    
    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class EncoderVGGTrans(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.backbone = model.basemodel.backbone
        # self.d_model = model.basemodel.transformerLM.d_model
        # self.encoder = model.basemodel.transformerLM.transformer.encoder
        # self.pos_enc = model.basemodel.transformerLM.pos_enc
        self.backbone = model.backbone
        self.d_model = model.transformerLM.d_model
        self.encoder = model.transformerLM.transformer.encoder
        self.pos_enc = model.transformerLM.pos_enc

    def forward(self, img):
        src = self.backbone(img)
        src = self.pos_enc(src*math.sqrt(self.d_model))
        memory = self.encoder(src)
        return memory

class DecoderVGGTrans(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.decoder = model.basemodel.transformerLM.transformer.decoder
        # self.d_model = model.basemodel.transformerLM.d_model
        # self.embed_tgt = model.basemodel.transformerLM.embed_tgt
        # self.pos_enc = model.basemodel.transformerLM.pos_enc
        # self.fc = model.basemodel.transformerLM.fc
        # self.rethinking = model.rethinking
        # self.fc_out = model.fc_out
        self.decoder = model.transformerLM.transformer.decoder
        self.d_model = model.transformerLM.d_model
        self.embed_tgt = model.transformerLM.embed_tgt
        self.pos_enc = model.transformerLM.pos_enc
        self.fc = model.transformerLM.fc

    def forward(self, tgt, memory, tgt_mask):
        # tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        de_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        de_out = self.fc(de_out.transpose(0, 1))

        # de_out = de_out + self.rethinking(de_out, tgt_mask)
        # return self.fc_out(de_out)
        return de_out

device="cuda:3"

def gen_nopeek_mask(length):
    mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    model = VGGTransformer(config)
    model.load_state_dict(torch.load("/mnt/disk4/VN_HTR/VN_HTR/VGGTransformer/weights/cp_vgg_transformer_finetune_v4.pt", map_location=device))
    model.to(device)

    #data loader
    # anot = pd.read_csv("concated_anot.csv").sample(frac=1).reset_index(drop=True)
    anot = pd.read_csv("test_anot.csv").sample(frac=1, random_state=0).reset_index(drop=True)

    valdataset = CustomDataset(anot, config)
    valloader = DataLoader(dataset=valdataset, batch_size=2, shuffle=False, generator=torch.Generator(device="cpu"))

    for batch in valloader:
        img = batch["image"].to(device)
        in_tokens = batch["tokens"][:,:-1].to(device)
        max_width = batch["width"].max().item()

        encoder = EncoderVGGTrans(model)
        encoder.eval()
        trace_encoder = torch.jit.trace(encoder, img[:, :, :, :max_width])
        trace_encoder.save("/mnt/disk4/VN_HTR/VN_HTR/vnhtr/weights/vt_encoder.pt")
        memory = encoder(img[:, :, :, :max_width])#.transpose(1, 0)
        # print(memory.shape)
        # print(in_tokens[:, :5].shape)

        decoder = DecoderVGGTrans(model)
        decoder.eval()
        trace_decoder = torch.jit.trace(decoder, (in_tokens[:, :3].transpose(0,1).to(device), memory, gen_nopeek_mask(in_tokens[:, :3].shape[1]).to(device)))
        trace_decoder.save("/mnt/disk4/VN_HTR/VN_HTR/vnhtr/weights/vt_decoder.pt")
        
        break

