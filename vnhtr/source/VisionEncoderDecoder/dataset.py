import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrOCRProcessor
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, anot, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        self.anot = anot
        self.max_seq_len = config.max_seq_len

    def __len__(self):
        return len(self.anot)

    def __getitem__(self, idx):
        sample = self.anot.iloc[idx]
        fn = sample.filename

        input_ids = self.tokenizer(str(sample.label), return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_seq_len)

        img_path = "augmented_images/" + fn
        if fn[:4] == "wild":
            img_path = "WildLine/" + fn
        if fn[:5] == "digit":
            img_path = "digits/" + fn
        if fn[:6] == "single":
            img_path = "single_digit/" + fn

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        return {
            "pixel_values":    pixel_values[0],
            "input_ids":   input_ids["input_ids"][0],
            "att_mask":     input_ids["attention_mask"][0]
        }