import torch
import numpy as np
from torch.nn import functional as F

def infer_transformer_base(img, net, max_seq_length=128, sos_token=1, eos_token=2, backbone="vgg", adapter=False):
    "data: BxCXHxW"
    net.eval()
    device = img.device
    net.to(device)
    if not adapter:
        model = net
    else:
        model = net.basemodel

    with torch.no_grad():
        src = model.backbone(img)
        if backbone != "vgg":
            src = src.transpose(1,0)
        
        memory = model.transformerLM.forward_encoder(src)
        
    
        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
            output, memory = model.transformerLM.forward_decoder(tgt_inp, memory)
            if adapter:
                output = net.forward_adapt(output)
            output = F.softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
    
    return translated_sentence

def param_to_update(net):
        encoder_param = []
        decoder_param = []
        for name, param in net.named_parameters():
            param = param.float()
            param.requires_grad = True
            if "backbone" in name:
                encoder_param.append(param)
            else:
                decoder_param.append(param)
        return encoder_param, decoder_param