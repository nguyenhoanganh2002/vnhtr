from model import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from evaluate import load
import wandb
from config import *


if __name__ == '__main__':
    #define model
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    model = VNTrOCR()
    model.load_state_dict(torch.load("source/weights/cp_vision_encoder_decoder_augmented_data.pt"))
    model.to(device)

    #data loader
    anot = pd.read_csv("concated_anot.csv").sample(frac=1, random_state=0).reset_index(drop=True)

    traindataset = CustomDataset(anot[:-10000], config)
    trainloader = DataLoader(dataset=traindataset, batch_size=16, shuffle=True, generator=torch.Generator(device="cpu"), num_workers=8)
    valdataset = CustomDataset(anot[-10000:], config)
    valloader = DataLoader(dataset=valdataset, batch_size=16, shuffle=False, generator=torch.Generator(device="cpu"), num_workers=8)

    #optimizer
    param1, param2 = param_to_update(model)
    optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.0001},
                                {'params': param2, 'lr': 0.0001}], weight_decay=1e-5)

    #loss function
    # lsLoss = LabelSmoothingLoss(233, padding_idx=0, smoothing=0.1)

    #utils
    cer = load("cer")

    #number of epochs
    n_epoch = 15
    #checkpoint
    best_cer = 1

    # wandb init
    # wandb.login(key = '9d97bfb2299059fecbd3e48dd349fafce5906d76')
    # wandb.login(key = 'e0e0a2547f255a36f551d7b6a166b84e5139d276')
    # wandb.init(
    #   project="VN_HTR",
    #   name=f"vision_encoder_decoder_augmented_data",
    #   config={
    #       "learning_rate": 0.0001,
    #       "architecture": "v_t",
    #       "dataset": "gen",
    #       "epochs": 10,
    #   }
    # )


    print("Start .....")
    for epoch in range(n_epoch):
        print(f"epoch {epoch} ----------------------------")
        if epoch < 1:
            optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.00005},
                            {'params': param2, 'lr': 0.00005}], weight_decay=1e-5)
        elif epoch < 3:
            optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.00001},
                            {'params': param2, 'lr': 0.00001}], weight_decay=1e-5)
        else:
            optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.000005},
                            {'params': param2, 'lr': 0.000005}], weight_decay=1e-5)
        model.train()
        train_loss = 0
        torch.cuda.empty_cache()
        for batch in tqdm(trainloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            att_mask = batch["att_mask"].to(device)

            optimizer.zero_grad()

            # forward
            loss = model(pixel_values, input_ids, att_mask)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
            
            optimizer.step()

            train_loss += loss.item()

        else:
            print(f"train loss: {train_loss/len(trainloader)}")
            # torch.save(model.state_dict(), "source/weights/last_vgg_transformer.pt")
            if True:
                with torch.no_grad():
                    model.eval()
                    predictions, references = [], []
                    torch.cuda.empty_cache()
                    for batch in tqdm(valloader):
                        pixel_values = batch["pixel_values"].to(device)
                        input_ids = batch["input_ids"].to(device)

                        preds = infer(pixel_values, model)

                        predictions += valdataset.tokenizer.batch_decode(preds, skip_special_tokens=True)
                        references += valdataset.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    for i in range(len(references)):
                        if references[i] == "":
                            references[i] = " "
                    cer_score = 1
                    try:
                        cer_score = cer.compute(predictions=predictions, references=references, concatenate_texts=True)
                    except:
                        print("error when caculate cer score")
                    # wandb.log({"train_loss": train_loss/len(trainloader), "cer": cer_score})
                    if cer_score < best_cer:
                        best_cer = cer_score
                        torch.save(model.state_dict(), "source/weights/cp_finetune.pt")

                    print(f'Character Error Rate: {cer_score}')
    
    # wandb.finish()