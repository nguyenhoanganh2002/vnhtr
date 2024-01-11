from dataclasses import dataclass
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch

device = torch.device("cuda:0")

@dataclass
class Config:
    n_decoder_blocks: int = 3
    n_encoder_blocks: int = 3
    im_height: int = 32
    im_width: int = 512
    n_embd: int = 256
    feat_depth: int = 256
    n_heads: int = 8
    dropout: float = 0.4
    att_dropout: float = 0.0
    bias: bool = False
    pe_dropout: float = 0.2
    pe_maxlen: int = 100000
    max_seq_len: int = 128
    in_chanel: int = 3
    vocab_size: int = 233

config = Config()

cfg = Cfg.load_config_from_name('vgg_transformer')
cfg['cnn']['pretrained']=False
cfg['device'] = device
predictor = Predictor(cfg).model