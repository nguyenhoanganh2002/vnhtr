from model import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from evaluate import load
import wandb
from config import *

device="cuda:2"

if __name__ == '__main__':
    #define model
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    # model = VNTrOCR()
    # model.load_state_dict(torch.load("VisionEncoderDecoder/weights/cp_finetune_v2_5.pt", map_location=device))
    model = AdapterVNTrOCR(config)
    model.load_state_dict(torch.load("VisionEncoderDecoder/weights/cp_add_finetune_v2.pt", map_location=device))
    model.to(device)
    
    valdataset = CustomDataset(pd.read_csv("test_anot.csv"), config)
    valloader = DataLoader(dataset=valdataset, batch_size=128, shuffle=False, generator=torch.Generator(device="cpu"), num_workers=8)


    #loss function
    # lsLoss = LabelSmoothingLoss(233, padding_idx=0, smoothing=0.1)

    #utils
    cer = load("cer")
    wer = load("wer")
    cer_score = 0
    wer_score = 0

    # wandb init
    # wandb.login(key = '9d97bfb2299059fecbd3e48dd349fafce5906d76')
    # wandb.login(key = 'e0e0a2547f255a36f551d7b6a166b84e5139d276')
    # wandb.init(
    #   project="fine-tune-VNHTR",
    #   name=f"vision_encoder_decoder_augmented_data",
    #   config={
    #       "learning_rate": 0.0001,
    #       "architecture": "v_t",
    #       "dataset": "gen",
    #       "epochs": 10,
    #   }
    # )


    print("Start .....")
    with torch.no_grad():
        model.eval()
        predictions, references = [], []
        # torch.cuda.empty_cache()
        for batch in tqdm(valloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            preds = infer(pixel_values, model, adapter=True)

            predictions += valdataset.tokenizer.batch_decode(preds, skip_special_tokens=True)
            references += valdataset.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        cer_score += cer.compute(predictions=predictions, references=references, concatenate_texts=True)
        wer_score += wer.compute(predictions=predictions, references=references, concatenate_texts=True)
        print(f'Character Error Rate: {cer_score}')
        print(f'Word Error Rate: {wer_score}')