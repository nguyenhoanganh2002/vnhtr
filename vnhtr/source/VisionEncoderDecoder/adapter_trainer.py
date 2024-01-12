from model import *
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from evaluate import load
import wandb
from config import *
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", 
                        type=int, 
                        default=os.getenv('LOCAL_RANK', -1), 
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = parser.parse_args()
    
    local_rank = args.local_rank
    torch.distributed.init_process_group(backend="nccl", rank = local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)

    #define model
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    model = AdapterVNTrOCR(config)
    # model = nn.DataParallel(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    #data loader
    anot = pd.read_csv("concated_anot.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    model.module.basemodel.load_state_dict(torch.load("VisionEncoderDecoder/weights/cp_vision_encoder_decoder_augmented_data.pt", map_location=device))

    traindataset = CustomDataset(anot[:-10000], config)
    train_sampler = DistributedSampler(dataset=traindataset)
    trainloader = DataLoader(dataset=traindataset, batch_size=16, generator=torch.Generator(device="cpu"), num_workers=2, sampler = train_sampler, pin_memory = True)
    valdataset = CustomDataset(anot[-10000:], config)
    valloader = DataLoader(dataset=valdataset, batch_size=96, generator=torch.Generator(device="cpu"), num_workers=2, pin_memory = True)

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
    # wandb.login(key = '')
    wandb.login(key = '')
    wandb.init(
      project="VN_HTR",
      name=f"adapter_vision_encoder_decoder_add",
      config={
          "learning_rate": 0.0001,
          "architecture": "v_t",
          "dataset": "gen",
          "epochs": 10,
      }
    )

    print(local_rank)
    print("Start .....")
    dist.barrier()
    torch.backends.cudnn.benchmark = True
    for epoch in range(n_epoch):
        dist.barrier()        
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        if epoch != 0:
            state_dict = torch.load('/mnt/disk4/VN_HTR/VN_HTR/VisionEncoderDecoder/new_check_point/tmp.pt', map_location = 'cuda:{}'.format(local_rank))
            model.module.load_state_dict(state_dict)
        
        print(f"epoch {epoch} ----------------------------")
        if epoch < 1:
            optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.00005},
                            {'params': param2, 'lr': 0.00005}], weight_decay=1e-5)
        # elif epoch < 3:
        #     optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.00001},
        #                     {'params': param2, 'lr': 0.00001}], weight_decay=1e-5)
        else:
            optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.00001},
                            {'params': param2, 'lr': 0.00001}], weight_decay=1e-5)
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
            dist.barrier()
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

                        preds = infer(pixel_values, model.module, adapter=True)

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
                    wandb.log({"train_loss": train_loss/len(trainloader), "cer": cer_score})
                    if local_rank == 0:
                        torch.save(model.module.state_dict(), "/mnt/disk4/VN_HTR/VN_HTR/VisionEncoderDecoder/new_check_point/tmp.pt")

                    if cer_score < best_cer and local_rank == 0:
                        best_cer = cer_score
                        torch.save(model.module.state_dict(), "VisionEncoderDecoder/weights/cp_adapter_add.pt")

                    print(f'Character Error Rate: {cer_score}')
    
    wandb.finish()