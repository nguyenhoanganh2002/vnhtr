from jit import TrOCR
import torch
from PIL import Image
from transformers import AutoTokenizer, TrOCRProcessor
import numpy as np

device = "cuda:2"

if __name__ == '__main__':
    predictor = TrOCR(device)
    torch.cuda.set_device(3)
    img = Image.open("/mnt/disk4/VN_HTR/VN_HTR/test/test11_test1.jpg")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    dummy_input = processor(img, return_tensors="pt").pixel_values.to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True, ), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    print("warming up ...")
    for _ in range(10):
        _ = predictor.predict(dummy_input)
    # MEASURE PERFORMANCE
    print("measuring ...")
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = predictor.predict(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)