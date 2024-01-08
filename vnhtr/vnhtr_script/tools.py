import torch
import numpy as np
from torch.nn import functional as F
import os
import tempfile
from PIL import Image, ImageOps
import gdown

class Tokenizer():
    def __init__(self):
        vocab = ['a', 'A', 'à', 'À', 'ả', 'Ả', 'ã', 'Ã', 'á', 'Á', 'ạ', 'Ạ', 'ă', 'Ă', 'ằ', 'Ằ', 'ẳ', 'Ẳ', 'ẵ', 'Ẵ', 'ắ', 'Ắ', 'ặ', 'Ặ', 'â', 'Â', 'ầ', 'Ầ', 'ẩ', 'Ẩ', 'ẫ', 'Ẫ', 'ấ', 'Ấ', 'ậ', 'Ậ', 'b', 'B', 'c', 'C', 'd', 'D', 'đ', 'Đ', 'e', 'E', 'è', 'È', 'ẻ', 'Ẻ', 'ẽ', 'Ẽ', 'é', 'É', 'ẹ', 'Ẹ', 'ê', 'Ê', 'ề', 'Ề', 'ể', 'Ể', 'ễ', 'Ễ', 'ế', 'Ế', 'ệ', 'Ệ', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'ì', 'Ì', 'ỉ', 'Ỉ', 'ĩ', 'Ĩ', 'í', 'Í', 'ị', 'Ị', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ò', 'Ò', 'ỏ', 'Ỏ', 'õ', 'Õ', 'ó', 'Ó', 'ọ', 'Ọ', 'ô', 'Ô', 'ồ', 'Ồ', 'ổ', 'Ổ', 'ỗ', 'Ỗ', 'ố', 'Ố', 'ộ', 'Ộ', 'ơ', 'Ơ', 'ờ', 'Ờ', 'ở', 'Ở', 'ỡ', 'Ỡ', 'ớ', 'Ớ', 'ợ', 'Ợ', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'ù', 'Ù', 'ủ', 'Ủ', 'ũ', 'Ũ', 'ú', 'Ú', 'ụ', 'Ụ', 'ư', 'Ư', 'ừ', 'Ừ', 'ử', 'Ử', 'ữ', 'Ữ', 'ứ', 'Ứ', 'ự', 'Ự', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'ỳ', 'Ỳ', 'ỷ', 'Ỷ', 'ỹ', 'Ỹ', 'ý', 'Ý', 'ỵ', 'Ỵ', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
        vocab = ["pad", "<sos>", "<eos>", "<unk>"] + vocab

        self.c_vocab = dict(zip(vocab, np.arange(len(vocab))))
        self.reverse_vocab = list(self.c_vocab.items())
        self.max_seq_len = 128

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

class VGGTransformer():
    def __init__(self, device="cuda:2"):
        encoder_path, decoder_path = self.download_weights()
        self.tkz = Tokenizer()
        self.encoder = torch.jit.load(encoder_path, map_location=device)#.to(device)
        self.decoder = torch.jit.load(decoder_path, map_location=device)#.to(device)
        print("Loaded weights")
        self.encoder.eval()
        self.decoder.eval()
        self.device = device
        self.warmup()

    def download_weights(self):
        encoder_path = os.path.join(tempfile.gettempdir(), "vta_encoder.pt")
        decoder_path = os.path.join(tempfile.gettempdir(), "vta_decoder.pt")
        if os.path.exists(encoder_path):
            print(f'VGG Transformer with Rethinking Head encoder weights {encoder_path} exsits. Ignore download!')
        else:
            print(f'Downloading VGG Transformer with Rethinking Head encoder weights {encoder_path} ...')
            gdown.download(id="179BRTjPBMQn5vd9Ah6DIklcox30zh4YQ", output=encoder_path, quiet=True)
            if not os.path.exists(encoder_path):
                print("Retrying ...")
                if os.system(f'curl -H "Authorization: Bearer ya29.a0AfB_byDhvrZegVfgDHMqnSXRhp763RCQjw7HhwBR-eN3DCTmx-Q6SlsXpd5QagdhICn2zy5Dpp8SVYRWiDuwhH-IemClyvaElQFiOQBYSL7Hxy_ddAEEv6HbuxbzcKKtvRnaXpvSOIJ8ui8g1O93iY0tFHO7hFLGfy9PaCgYKAYASARMSFQHGX2MiAwlU1xpS3ZceA0-121l89w0171" https://www.googleapis.com/drive/v3/files/179BRTjPBMQn5vd9Ah6DIklcox30zh4YQ?alt=media -o {encoder_path}'):
                    raise RuntimeError('Download encoder failed!')
            
        if os.path.exists(decoder_path):
            print(f'VGG Transformer with Rethinking Head decoder weights {decoder_path} exsits. Ignore download!')
        else:
            print(f'Downloading VGG Transformer with Rethinking Head decoder weights {decoder_path} ...')
            
            gdown.download(id="1dNJrjBF-FcjQgzckr4CKJyFdaLRuse6q", output=decoder_path, quiet=True)
            if not os.path.exists(decoder_path):
                print("Retrying ...")
                if os.system(f'curl -H "Authorization: Bearer ya29.a0AfB_byDhvrZegVfgDHMqnSXRhp763RCQjw7HhwBR-eN3DCTmx-Q6SlsXpd5QagdhICn2zy5Dpp8SVYRWiDuwhH-IemClyvaElQFiOQBYSL7Hxy_ddAEEv6HbuxbzcKKtvRnaXpvSOIJ8ui8g1O93iY0tFHO7hFLGfy9PaCgYKAYASARMSFQHGX2MiAwlU1xpS3ZceA0-121l89w0171" https://www.googleapis.com/drive/v3/files/1dNJrjBF-FcjQgzckr4CKJyFdaLRuse6q?alt=media -o {decoder_path}'):
                    raise RuntimeError('Download decoder failed!')
        return encoder_path, decoder_path

    def warmup(self):
        print("Warming up ...")
        img = torch.rand(2,3,32,768)
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                mem = self.encoder(img.to(self.device))
                translated_sentence = [[1]*len(img)]
                max_length = 0

                while max_length <= 128 and not all(np.any(np.asarray(translated_sentence).T==2, axis=1)):

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
    
    def predict(self, imgs, max_seq_length=128, sos_token=1, eos_token=2):
        print("Preprocessing ...")
        max_W = 0
        ims = []
        for img in imgs:
            im, W = self.process_input(img)
            ims.append(im)
            if max_W < W:
                max_W = W
        imgs = torch.FloatTensor(np.array(ims))[:,:,:,:max_W]

        print("Predicting ...")
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                mem = self.encoder(imgs.to(self.device))
                translated_sentence = [[sos_token]*len(imgs)]
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
            
                return [self.tkz.reverse_tokens(tokens) for tokens in translated_sentence]
    
    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def process_input(self, image):
        img = image.convert('RGB')
        image_height = 32

        w, h = img.size
        new_w = int(image_height*w/h)

        img = img.resize((new_w, image_height), Image.LANCZOS)
        img = ImageOps.expand(img, border=(0, 0, 768 - new_w, 0), fill='white')
        img = np.asarray(img).transpose(2,0, 1)
        img = img/255
        return img, new_w