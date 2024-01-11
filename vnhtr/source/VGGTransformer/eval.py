from models import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from evaluate import load
import wandb
import math
from jit import OCRModel

device="cuda:1"

if __name__ == '__main__':
    #define model
    torch.set_default_dtype(torch.float32)
    model = AdapterVGGTransformer(config)
    model.load_state_dict(torch.load("/mnt/disk4/VN_HTR/VN_HTR/source/weights/finetune_adapter_vgg_transformer_v3.pt", map_location=device))
    # model = VGGTransformer(config)
    # model.load_state_dict(torch.load("/mnt/disk4/VN_HTR/VN_HTR/source/weights/cp_vgg_transformer_finetune_v4.pt", map_location=device))
    model.to(device)
    # model = OCRModel(device=device)
    # print(model.encoder)
    # print(model.decoder)

    #data loader
    # anot = pd.read_csv("concated_anot.csv").sample(frac=1).reset_index(drop=True)
    
    anot = pd.read_csv("test_anot.csv")

    valdataset = CustomDataset(anot, config)
    valloader = DataLoader(dataset=valdataset, batch_size=128, shuffle=False, generator=torch.Generator(device="cpu"), num_workers=8)

    #utils
    cer = load("cer")
    wer = load("wer")
    cer_score = 0
    wer_score = -0.1
    tkz = Tokenizer(config)

    print("Starting ....")

    with torch.no_grad():
        # model.eval()
        predictions, references = [], []
        torch.cuda.empty_cache()
        for batch in tqdm(valloader):
            img = batch["image"].to(device)
            tokens = batch["tokens"][:,1:]
            max_width = batch["width"].max().item()

            t_preds = infer_transformer_base(img[:, :, :, :max_width], model.to(device), backbone="vgg", adapter=True)
            # t_preds = model.predict(img[:, :, :, :max_width])

            chars_true = [tkz.reverse_tokens(token) for token in tokens]
            chars_pred = [tkz.reverse_tokens(t_pred) for t_pred in t_preds]

            predictions += chars_pred
            references += chars_true

        cer_score += cer.compute(predictions=predictions, references=references, concatenate_texts=True)
        wer_score += wer.compute(predictions=predictions, references=references, concatenate_texts=True)
        print(f'Character Error Rate: {cer_score}')
        print(f'Word Error Rate: {wer_score}')