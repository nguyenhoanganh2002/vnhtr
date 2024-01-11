import numpy as np
import math
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from config import Config, device

config = Config()

class Tokenizer():
    def __init__(self, config):
        vocab = ['a', 'A', 'à', 'À', 'ả', 'Ả', 'ã', 'Ã', 'á', 'Á', 'ạ', 'Ạ', 'ă', 'Ă', 'ằ', 'Ằ', 'ẳ', 'Ẳ', 'ẵ', 'Ẵ', 'ắ', 'Ắ', 'ặ', 'Ặ', 'â', 'Â', 'ầ', 'Ầ', 'ẩ', 'Ẩ', 'ẫ', 'Ẫ', 'ấ', 'Ấ', 'ậ', 'Ậ', 'b', 'B', 'c', 'C', 'd', 'D', 'đ', 'Đ', 'e', 'E', 'è', 'È', 'ẻ', 'Ẻ', 'ẽ', 'Ẽ', 'é', 'É', 'ẹ', 'Ẹ', 'ê', 'Ê', 'ề', 'Ề', 'ể', 'Ể', 'ễ', 'Ễ', 'ế', 'Ế', 'ệ', 'Ệ', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'ì', 'Ì', 'ỉ', 'Ỉ', 'ĩ', 'Ĩ', 'í', 'Í', 'ị', 'Ị', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ò', 'Ò', 'ỏ', 'Ỏ', 'õ', 'Õ', 'ó', 'Ó', 'ọ', 'Ọ', 'ô', 'Ô', 'ồ', 'Ồ', 'ổ', 'Ổ', 'ỗ', 'Ỗ', 'ố', 'Ố', 'ộ', 'Ộ', 'ơ', 'Ơ', 'ờ', 'Ờ', 'ở', 'Ở', 'ỡ', 'Ỡ', 'ớ', 'Ớ', 'ợ', 'Ợ', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'ù', 'Ù', 'ủ', 'Ủ', 'ũ', 'Ũ', 'ú', 'Ú', 'ụ', 'Ụ', 'ư', 'Ư', 'ừ', 'Ừ', 'ử', 'Ử', 'ữ', 'Ữ', 'ứ', 'Ứ', 'ự', 'Ự', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'ỳ', 'Ỳ', 'ỷ', 'Ỷ', 'ỹ', 'Ỹ', 'ý', 'Ý', 'ỵ', 'Ỵ', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
        vocab = ["pad", "<sos>", "<eos>", "<unk>"] + vocab

        self.c_vocab = dict(zip(vocab, np.arange(len(vocab))))
        self.reverse_vocab = list(self.c_vocab.items())
        self.max_seq_len = config.max_seq_len

    def tokenize(self, seq):
        res = []
        for c in ['<sos>'] + seq + ['<eos>']:
            try:
                res.append(self.c_vocab[c])
            except:
                res.append(self.c_vocab["<unk>"])
        mask = [1]*len(res)
        if len(res) < self.max_seq_len:
            n_pad = self.max_seq_len-len(res)+1
            res += [self.c_vocab["pad"]]*n_pad
            mask += [0]*n_pad
        elif len(res) > self.max_seq_len:
            s_pad = self.max_seq_len
            res = res[:s_pad] + [self.c_vocab["<eos>"]]
            mask = mask[:s_pad+1]
        else:
            res += [self.c_vocab["pad"]]
            mask += [0]
        return np.array(res), np.logical_not(mask)

    def reverse_tokens(self, tokens):
        res = []
        for token in tokens:
            if token in [self.c_vocab["<unk>"], self.c_vocab["pad"], self.c_vocab["<sos>"]]:
                continue
            if token == self.c_vocab["<eos>"]:
                break
            res.append(self.reverse_vocab[token][0])
            
        return ''.join(res)

    def reverse_tokens_special(self, tokens):
        res = []
        for token in tokens:
            res.append(self.reverse_vocab[token][0])
            # if token == self.c_vocab["<eos>"]:
            #     break
            
        return res

def process_input(image):
    img = image.convert('RGB')
    image_height = config.im_height

    w, h = img.size
    new_w = int(image_height*w/h)

    img = img.resize((new_w, image_height), Image.LANCZOS)
    img = ImageOps.expand(img, border=(0, 0, config.im_width - new_w, 0), fill='white')
    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img, new_w

class CustomDataset(Dataset):
    def __init__(self, anot, config):
        super().__init__()
        self.gen_source = "images_gen/"
        self.grey_source = "grey/"
        self.word_source = "grey_word/"
        self.tokenizer = Tokenizer(config)
        self.anot = anot

    def __len__(self):
        return len(self.anot)

    def __getitem__(self, idx):
        sample = self.anot.iloc[idx]
        fn = sample.filename
        # if fn[0] == 'g':
        #     img_path = self.gen_source + fn
        # elif fn[0] == 'w':
        #     img_path = self.word_source + fn
        # else:
        #     img_path = self.grey_source + fn
        
        chars = list(sample.label)
        tokens, mask = self.tokenizer.tokenize(chars)

        img_path = "augmented_images/" + fn
        if fn[:4] == "wild":
            img_path = "WildLine/" + fn
        if fn[:5] == "digit":
            img_path = "digits/" + fn
        if fn[:6] == "single":
            img_path = "single_digit/" + fn

        image = Image.open(img_path)
        image, width = process_input(image)

        return {
            "image":    torch.FloatTensor(image).to("cpu"),
            "tokens":   torch.LongTensor(tokens).to("cpu"),
            "padding_mask":     torch.FloatTensor(mask).to("cpu"),
            "width": torch.LongTensor([width]).to("cpu")
        }