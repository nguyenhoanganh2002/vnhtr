# Vietnamese Handwriting Text Recognition (aka vnhtr package)

This project deploys and improves two foundational models within [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) and [VietOCR](https://github.com/pbcquoc/vietocr).

## Proposal Architecture
### VGG Transformer with Rethinking Head
![VGG Transformer with Rethinking Head](https://github.com/nguyenhoanganh2002/vnhtr/assets/79850337/82876cdd-b84a-47da-9339-6362bd0400d1)
### TrOCR with Rethinking Head
![TrOCR with Rethinking Head](https://github.com/nguyenhoanganh2002/vnhtr/assets/79850337/9295c94f-5059-4a03-a3f3-950e0ab92e30)
## Usage
### `vnhtr` package
```bash
pip install vnhtr
```
```python
from PIL import Image
from vnhtr.vnhtr_script.tools import *

vta_predictor = VGGTransformer("cuda:0")
tra_predictor = TrOCR("cuda:0")

vta_predictor.predict([Image.open("/content/out_sample_2.jpg")])
tra_predictor.predict([Image.open("/content/out_sample_2.jpg")])
```
### Fully implemented
```bash
git clone https://github.com/nguyenhoanganh2002/vnhtr
cd ./vnhtr/vnhtr/source
pip install -r requirements.txt
```
* Pretrain/Fintune VGG Transformer/TrOCR (pretraining on a large dataset and then finetuning on a wild dataset) 
```bash
python VGGTransformer/train.py
python VisionEncoderDecoder/train.py
```
* Pretrain VGG Transformer/TrOCR with Rethinking Head (large dataset)
```bash
python VGGTransformer/adapter_trainer.py
python VisionEncoderDecoder/adapter_trainer.py
```
* Finetune VGG Transformer with Rethinking Head (wild dataset)
```bash
python VGGTransformer/finetune.py
python VisionEncoderDecoder/finetune.py
```
* Access the model without going through the training or finetuning phases.
```python
from VGGTransformer.config import config as vggtransformer_cf
from VGGTransformer.models import VGGTransformer, AdapterVGGTransformer
from VisionEncoderDecoder.config import config as trocr_cf
from VisionEncoderDecoder.model import VNTrOCR, AdapterVNTrOCR

vt_base = VGGTransformer(vggtransformer_cf)
vt_adapter = AdapterVGGTransformer(vggtransformer_cf)
tr_base = VNTrOCR(trocr_cf)
tr_adapter = AdapterVNTrOCR(trocr_cf)
```

For access to the full dataset and pretrained weights, please contact: [anh.nh204511@gmail.com](mailto:anh.nh204511@gmail.com)
