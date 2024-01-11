import os
import random
import numpy as np
from tqdm import  tqdm


def rand_parts(seq, n):
    indices = range(len(seq) - (8 - 1) * n)
    result = []
    offset = 0
    ls = np.random.poisson(6, n)
    j = 0
    for i in tqdm(sorted(random.sample(indices, n))):
        l = max(2, min(ls[j], 12))
        j += 1
        i += offset
        try:
            result.append(" ".join(seq[i:i+l]))
        except:
            result.append(" ".join(seq[i:]))
            break
        offset += l - 1
    return result

def process_folder(path):
    fns = os.listdir(path)
    res = []
    for fn in tqdm(fns):
        f = open(path + "/" + fn, 'r', encoding='utf-16')
        content = f.read()
        f.close()
        ignore_char = ['!', '"', '%', '&', '(', ')', '*', ',', '-', '.', '/', ':', ';', '?']
        for char in ignore_char:
            content = content.replace(char, ' ')
        sentences = content.split()
        res += rand_parts(sentences, round(len(sentences)/10))

    with open("sentences.txt", 'a', encoding='utf-16') as f:
        f.write("\n".join(res) + "\n")
        f.close()

if __name__ == "__main__":
    # for folder in os.listdir("VNTC"):
    #     process_folder("VNTC/" + folder)

    res = []
    f = open("corpus-title.txt", 'r', encoding='utf-8')
    content = f.read()
    f.close()
    ignore_char = ['!', '"', '%', '&', '(', ')', '*', ',', '-', '.', '/', ':', ';', '?']
    for char in ignore_char:
        content = content.replace(char, ' ')
    sentences = content.split()
    try:
        res += rand_parts(sentences, round(len(sentences) / 8))
    except:
        pass
    with open("sentences.txt", 'a', encoding='utf-16') as f:
        f.write("\n".join(res) + "\n")
        f.close()
