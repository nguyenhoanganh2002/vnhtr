import torch
import torch.nn as nn
from torch.nn import functional as F

def infer(pixel_values, net, max_seq_length=35, sos_token=0, eos_token=2, adapter=False):
    "data: BxCXHxW"
    net.eval()
    device = pixel_values.device
    net.to(device)
    if not adapter:
        model = net
    else:
        model = net.basemodel

    with torch.no_grad():
        encoder_output = model.forward_encode(pixel_values)
    
        start_ids = torch.LongTensor([[sos_token]]*pixel_values.shape[0]).to(device)

        max_length = 0

        while max_length <= max_seq_length and not all(start_ids[:, -1] == eos_token):
            output = model.forward_decode(start_ids, encoder_output)
            if adapter:
                output = net.forward_adapt(output)
            output = F.softmax(output[:,-1,:], dim=-1)

            start_ids = torch.cat([start_ids, output.argmax(dim=-1).unsqueeze(1)], dim=-1)

            max_length += 1
    
    return start_ids

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