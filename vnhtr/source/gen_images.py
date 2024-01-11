import json
import cv2
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

with open("word_anot.json", encoding='utf-8') as fh:
    data = json.load(fh)

vocab = {}
for word in tqdm(data.keys()):
    fns = data[word]
    w_im = [cv2.imread("grey_word/w_" + fn).astype(np.int32) for fn in fns]
    vocab[word] = w_im

def resize_n_pad(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # d_h_size = 64
    # max_w = 1024
    #
    # old_size = im.shape[:2]  # old_size is in (height, width) format
    #
    # ratio = 1.0 * d_h_size / old_size[0]
    # new_size = [int(x * ratio) for x in old_size]
    # new_size[1] = min(max_w, new_size[1])
    # new_size[0] = 64
    #
    # # new_size should be in (width, height) format
    # im = cv2.resize(im, (new_size[1], new_size[0]))
    #
    # delta_w = max_w - new_size[1]
    #
    # new_im = cv2.copyMakeBorder(im, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=255)
    return im.astype(np.uint8)

def gen_image(sub_sentence):
    words = sub_sentence.split()
    ims = []
    label = []
    for word in words:
        try:
            image = random.choice(vocab[word])
        except:
            continue
        if image.shape[0] == 63:
            image = np.concatenate((image, image[0:1,:,:]), axis=0)
        image += (random.choice(np.arange(140,171))-np.mean(image).astype(np.int32))
        ims.append(image)
        label.append(word)
    # cv2.imshow("test", np.concatenate(ims, axis=1).astype(np.uint8))
    # cv2.waitKey()
    im = np.concatenate(ims, axis=1).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.uint8), " ".join(label)

def get_content(sub_sentence):
    words = sub_sentence.split()
    contents = []
    for word in words:
        try:
            image = random.choice(vocab[word])
            contents.append(word)
        except:
            continue
    return " ".join(contents)


if __name__ == "__main__":
    f = open("sentences.txt", 'r', encoding='utf-16')
    contents = f.read().split("\n")
    f.close()
    anot_json = {"filename": [], "label": []}
    for i, content in tqdm(enumerate(contents[:5000000])):
        try:
            new_image, label = gen_image(content)
        except:
            continue
        if len(label) != 0:
            anot_json["filename"].append(f"gen_{i}.jpg")
            anot_json["label"].append(label)
            cv2.imwrite(f"images_gen/gen_{i}.jpg", new_image)
        if i % 100000 == 0:
            pd.DataFrame(data=anot_json).to_csv("gen_anot.csv", index=False)
    pd.DataFrame(data=anot_json).to_csv("gen_anot.csv", index=False)