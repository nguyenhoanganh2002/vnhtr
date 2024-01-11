from dataclasses import dataclass
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch

# device = torch.device("cuda")
device="cuda:3"

@dataclass
class Config:
    n_decoder_blocks: int = 6
    n_encoder_blocks: int = 12
    im_height: int = 384
    im_width: int = 384
    n_embd: int = 768
    max_seq_len: int = 35
    in_chanel: int = 3
    rank: int = 16
    vocab_size: int = 40030
    bias: bool = False

config = Config()
