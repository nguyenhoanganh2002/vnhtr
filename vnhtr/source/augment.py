import cv2
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

def add_line(img):

    h, w, c = img.shape
    line_mask = np.ones((h,w,c), dtype=np.uint8) * 255

    # line
    pos = np.random.uniform(0.72, 0.78)
    start = (0, int(pos*img.shape[0]))
    end = (img.shape[1], int(pos*img.shape[0]))

    # draw line
    # mean_rgb = int(0.3*img.mean())
    mean_rgb = int(np.random.uniform(0.25, 0.45)*img.mean())
    line_mask = cv2.line(line_mask,start,end,(mean_rgb,mean_rgb,mean_rgb),2)

    # blur line
    line_mask = cv2.GaussianBlur(line_mask, (7, 7), 0)

    # merge line to raw image
    img = np.minimum(img, line_mask)

    return img

def add_dotted_line(img):

    h, w, c = img.shape
    line_mask = np.ones((h,w,c), dtype=np.uint8) * 255

    # line
    pos = np.random.uniform(0.72, 0.78)
    start = (0, int(pos*img.shape[0]))
    end = (img.shape[1], int(pos*img.shape[0]))

    # draw line
    mean_rgb = np.random.randint(0, 10)
    line_mask = cv2.line(line_mask,start,end,(mean_rgb,mean_rgb,mean_rgb), 2)
    delete_idxs = []
    blank = np.random.randint(2, 7)
    for j in range(blank):
        delete_idxs += [i for i in range(j, w, 4+blank-2)]

    line_mask[start[1]-1, delete_idxs, :] = [[255,255,255]]*len(delete_idxs)
    line_mask[start[1], delete_idxs, :] = [[255,255,255]]*len(delete_idxs)
    line_mask[start[1]+1, delete_idxs, :] = [[255,255,255]]*len(delete_idxs)

    # blur line
    line_mask = cv2.GaussianBlur(line_mask, (5, 5), 0)

    # merge line to raw image
    img = np.minimum(img, line_mask)

    return img

def add_grid(img):

    h, w, c = img.shape

    mean_rgb = int(np.random.uniform(0.25, 0.45)*img.mean())

    # draw row
    pos1 = np.random.uniform(0, 0.06)
    rstart = [(0, int((i+pos1)*img.shape[0])) for i in [0, 0.25, 0.5, 0.75]]
    rend = [(img.shape[1], int((i+pos1)*img.shape[0])) for i in [0, 0.25, 0.5, 0.75]]
    row_mask = np.ones((h,w,c), dtype=np.uint8) * 255
    for i in range(4):
        row_mask = cv2.line(row_mask,rstart[i],rend[i],(mean_rgb,mean_rgb,mean_rgb),2)
    row_mask = cv2.GaussianBlur(row_mask, (7, 7), 0)

    #draw column
    pos2 = np.random.randint(0,5)
    cstart = [(i, 0) for i in range(pos2, w, 16)]
    cend = [(i, h) for i in range(pos2, w, 16)]
    column_mask = np.ones((h,w,c), dtype=np.uint8) * 255
    for i in range(len(cstart)):
        column_mask = cv2.line(column_mask,cstart[i],cend[i],(mean_rgb,mean_rgb,mean_rgb),2)
    column_mask = cv2.GaussianBlur(column_mask, (7, 7), 0)

    grid_mask = np.minimum(row_mask, column_mask)

    # merge line to raw image
    img = np.minimum(img, grid_mask)

    return img

def get_path(filename):
    s_char = filename[0]
    if s_char == 'g':
        return "images_gen/" + filename
    if s_char == 'w':
        return "grey_word/" + filename
    return "grey/" + filename

def augment():
    print("Start ...")
    anot = pd.read_csv("adjust_anot.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    # anot.to_csv("augmented_anot.csv", index=False)
    anot = anot.to_dict('records')
    l_anot = anot[:1000000]
    d_anot = anot[1000000:2000000]
    g_anot = anot[2000000:3000000]
    _anot = anot[3000000:]
    print("1--------------------")
    for i in tqdm(range(len(l_anot))):
        fn = l_anot[i]["filename"]
        img = cv2.imread(get_path(fn))
        cv2.imwrite("augmented_images/"+fn, add_line(img))
    
    print("2--------------------")
    for i in tqdm(range(len(d_anot))):
        fn = d_anot[i]["filename"]
        img = cv2.imread(get_path(fn))
        cv2.imwrite("augmented_images/"+fn, add_dotted_line(img))
    
    print("3--------------------")
    for i in tqdm(range(len(g_anot))):
        fn = g_anot[i]["filename"]
        img = cv2.imread(get_path(fn))
        cv2.imwrite("augmented_images/"+fn, add_grid(img))
    
    print("4--------------------")
    for i in tqdm(range(len(_anot))):
        fn = _anot[i]["filename"]
        img = cv2.imread(get_path(fn))
        cv2.imwrite("augmented_images/"+fn, img)
    
    anot.sample(frac=1, random_state=0).reset_index(drop=True).to_csv("augmented_anot.csv", index=False)

if __name__ == "__main__":
    augment()
        
