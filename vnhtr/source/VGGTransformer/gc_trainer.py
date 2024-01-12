from models import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from evaluate import load
import wandb
import math

# device="cuda:1"

if __name__ == '__main__':
    #define model
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    model = GCTransformer(config)
    # model.load_state_dict(torch.load("source/weights/cp_GC_transformer.pt"))
    model.to(device)

    #data loader
    anot = pd.read_csv("test_anot.csv").sample(frac=1, random_state=42).reset_index(drop=True)

    traindataset = CustomDataset(anot[:1000], config)
    trainloader = DataLoader(dataset=traindataset, batch_size=24, shuffle=True, generator=torch.Generator(device="cpu"))
    valdataset = CustomDataset(anot[-1000:], config)
    valloader = DataLoader(dataset=valdataset, batch_size=24, shuffle=False, generator=torch.Generator(device="cpu"))

    #optimizer
    param1, param2 = param_to_update(model)
    optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.0001},
                            {'params': param2, 'lr': 0.0001}], weight_decay=1e-5)

    #loss function
    lsLoss = LabelSmoothingLoss(233, padding_idx=0, smoothing=0.1)

    #utils
    cer = load("cer")
    tkz = Tokenizer(config)

    #number of epochs
    n_epoch = 10
    #checkpoint
    best_cer = 1

    #wandb init
    # wandb.login(key = '')
    # wandb.init(
    #   project="VN_HTR",
    #   name=f"1M_GC_transformers",
    #   config={
    #       "learning_rate": 0.0001,
    #       "architecture": "v_t",
    #       "dataset": "gen",
    #       "epochs": 10,
    #   }
    # )

    max_len = {1: 8, 2: 16, 3: 21, 4: 26, 5: 33, 6: 38, 7: 43, 8: 49, 9: 54, 10: 64, 11: 60, 12: 63, 13: 66, 14: 66, 15: 66, 16: 70, 17: 80, 18: 90, 19:108, 20: 106, 21: 114}

    for epoch in range(n_epoch):
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
            img = batch["image"].to(device)
            in_tokens = batch["tokens"][:,:-1].to(device)
            padding_mask = batch["padding_mask"][:,:-1].to(device)
            max_width = batch["width"].max().item()
            max_token = min(max_len[min(math.ceil(max_width/config.im_height),21)] + 5, config.max_seq_len)

            optimizer.zero_grad()

            # forward
            outs = model(img[:, :, :, :max_width], in_tokens[:, :max_token], padding_mask[:, :max_token])

            # ground truth
            tokens = batch["tokens"][:,1:].to(device)

            loss = lsLoss(outs, tokens[:, :max_token])
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
            
            optimizer.step()

            train_loss += loss.item()

        else:
            print(f"train loss: {train_loss/len(trainloader)}")
            # torch.save(model.state_dict(), "source/weights/last_GC_transformer.pt")
            if True:
                with torch.no_grad():
                    model.eval()
                    predictions, references = [], []
                    torch.cuda.empty_cache()
                    for batch in tqdm(valloader):
                        img = batch["image"].to(device)
                        tokens = batch["tokens"][:,1:].to(device)
                        max_width = batch["width"].max().item()

                        t_preds = infer_transformer_base(img[:, :, :, :max_width], model.to(device), backbone="GC")

                        chars_true = [tkz.reverse_tokens(token) for token in tokens]
                        chars_pred = [tkz.reverse_tokens(t_pred) for t_pred in t_preds]

                        predictions += chars_pred
                        references += chars_true

                    cer_score = cer.compute(predictions=predictions, references=references, concatenate_texts=True)
                    # wandb.log({"train_loss": train_loss/len(trainloader), "cer": cer_score})
                    if cer_score < best_cer:
                        best_cer = cer_score
                        torch.save(model.state_dict(), "source/weights/GC_transformer.pt")

                    print(f'Character Error Rate: {cer_score}')
    
    # wandb.finish()