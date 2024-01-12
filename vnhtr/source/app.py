# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from transformers import AutoTokenizer, TrOCRProcessor
import torch
from VGGTransformer.jit import OCRModel
from VisionEncoderDecoder.jit import TrOCR
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

device = "cuda:2"

def process_input(image):
    img = image.convert('RGB')
    image_height = 32

    w, h = img.size
    new_w = int(image_height*w/h)

    img = img.resize((new_w, image_height), Image.LANCZOS)
    img = ImageOps.expand(img, border=(0, 0, 768 - new_w, 0), fill='white')
    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img, new_w

class Tokenizer():
    def __init__(self):
        vocab = ['a', 'A', 'à', 'À', 'ả', 'Ả', 'ã', 'Ã', 'á', 'Á', 'ạ', 'Ạ', 'ă', 'Ă', 'ằ', 'Ằ', 'ẳ', 'Ẳ', 'ẵ', 'Ẵ', 'ắ', 'Ắ', 'ặ', 'Ặ', 'â', 'Â', 'ầ', 'Ầ', 'ẩ', 'Ẩ', 'ẫ', 'Ẫ', 'ấ', 'Ấ', 'ậ', 'Ậ', 'b', 'B', 'c', 'C', 'd', 'D', 'đ', 'Đ', 'e', 'E', 'è', 'È', 'ẻ', 'Ẻ', 'ẽ', 'Ẽ', 'é', 'É', 'ẹ', 'Ẹ', 'ê', 'Ê', 'ề', 'Ề', 'ể', 'Ể', 'ễ', 'Ễ', 'ế', 'Ế', 'ệ', 'Ệ', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'ì', 'Ì', 'ỉ', 'Ỉ', 'ĩ', 'Ĩ', 'í', 'Í', 'ị', 'Ị', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ò', 'Ò', 'ỏ', 'Ỏ', 'õ', 'Õ', 'ó', 'Ó', 'ọ', 'Ọ', 'ô', 'Ô', 'ồ', 'Ồ', 'ổ', 'Ổ', 'ỗ', 'Ỗ', 'ố', 'Ố', 'ộ', 'Ộ', 'ơ', 'Ơ', 'ờ', 'Ờ', 'ở', 'Ở', 'ỡ', 'Ỡ', 'ớ', 'Ớ', 'ợ', 'Ợ', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'ù', 'Ù', 'ủ', 'Ủ', 'ũ', 'Ũ', 'ú', 'Ú', 'ụ', 'Ụ', 'ư', 'Ư', 'ừ', 'Ừ', 'ử', 'Ử', 'ữ', 'Ữ', 'ứ', 'Ứ', 'ự', 'Ự', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'ỳ', 'Ỳ', 'ỷ', 'Ỷ', 'ỹ', 'Ỹ', 'ý', 'Ý', 'ỵ', 'Ỵ', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
        vocab = ["pad", "<sos>", "<eos>", "<unk>"] + vocab

        self.c_vocab = dict(zip(vocab, np.arange(len(vocab))))
        self.reverse_vocab = list(self.c_vocab.items())
        self.max_seq_len = 128

    def tokenize(self, seq):
        res = []
        for c in ['<sos>'] + seq + ['<eos>']:
            try:
                res.append(self.c_vocab[c])
            except:
                res.append(self.c_vocab["<unk>"])
        mask = [1]*len(res)
        if len(res) < self.max_seq_len:
            n_pad = self.max_seq_len-len(res)+1
            res += [self.c_vocab["pad"]]*n_pad
            mask += [0]*n_pad
        elif len(res) > self.max_seq_len:
            s_pad = self.max_seq_len
            res = res[:s_pad] + [self.c_vocab["<eos>"]]
            mask = mask[:s_pad+1]
        else:
            res += [self.c_vocab["pad"]]
            mask += [0]
        return np.array(res), np.logical_not(mask)

    def reverse_tokens(self, tokens):
        res = []
        for token in tokens:
            if token in [self.c_vocab["<unk>"], self.c_vocab["pad"], self.c_vocab["<sos>"]]:
                continue
            if token == self.c_vocab["<eos>"]:
                break
            res.append(self.reverse_vocab[token][0])
            
        return ''.join(res)

    def reverse_tokens_special(self, tokens):
        res = []
        for token in tokens:
            res.append(self.reverse_vocab[token][0])
            # if token == self.c_vocab["<eos>"]:
            #     break
            
        return res

if 'session_state' not in st.session_state:
    st.session_state.session_state = {
        'v_result': None,
        'va_result': None,
        'tr_result': None,
        'tra_result': None,
    }

def perform_inference(img, model, label):
    pred, token, prob, conflict = None, None, None, None
    
    if model is not None:
        pred, token, prob, conflict = model.predict_topk(img)
        st.session_state[f'{label.lower().replace(" ", "_")}_result'] = {
            'pred': pred,
            'token': token,
            'prob': prob,
            'conflict': conflict,
        }
        if label == "VGG Transformer":
            st.write("VGG Transformer:\t\t\t" + tkz.reverse_tokens(pred[0]))
        elif label == "VGG Transformer with Rethinking Head":
            st.write("VGG Transformer with Rethinking Head:\t" + tkz.reverse_tokens(pred[0]))
        elif label == "TrOCR":
            st.write("TrOCR:\t\t\t\t\t" + tokenizer.batch_decode(pred, skip_special_tokens=True)[0])
        else:st.write("TrOCR with Rethinking Head:\t\t" + tokenizer.batch_decode(pred, skip_special_tokens=True)[0])


    return {
        'pred': pred,
        'token': token,
        'prob': prob,
        'conflict': conflict,
    }

def inference(img, opt):
    img1, W = process_input(img)
    img1 = torch.FloatTensor(img1[:,:W,:]).unsqueeze(0).to(device)
    img2 = processor(img, return_tensors="pt").pixel_values.to(device)
    
    if opt == 0:
        st.session_state.v_result = perform_inference(img1, vgg_base, "VGG Transformer")
    elif opt == 1:
        st.session_state.va_result = perform_inference(img1, vgg_adapter, "VGG Transformer with Rethinking Head")
    elif opt == 2:
        st.session_state.tr_result = perform_inference(img2, trocr_base, "TrOCR")
    else:
        st.session_state.tra_result = perform_inference(img2, trocr_base, "TrOCR with Rethinking Head")


def plot_entire(token, prob, conflict, topk=5):
    if conflict is not None:
        if conflict.shape[-1] < 500:
            weights = prob[:topk, :]
            token = token[:topk, :]

            fig, ax = plt.subplots(figsize=(20, 2.5))

            # Create a custom colormap that transitions from white to 'viridis'
            viridis_white = plt.cm.Blues(np.linspace(0, 1, 256))
            viridis_white[:, :3] = 1 - (1 - viridis_white[:, :3]) * 0.7  # Adjust the brightness

            # Create a LinearSegmentedColormap using the custom colormap
            cmap = LinearSegmentedColormap.from_list('custom_cmap', viridis_white)

            # Set the normalization to map the weights to the range [0, 1]
            norm = Normalize(vmin=0, vmax=1)

            # Use the created colormap in imshow with the specified norm
            cax = ax.imshow(weights, cmap=cmap, interpolation='none', norm=norm)

            for i in range(token.shape[0]):
                text = tkz.reverse_tokens_special(token[i])
                for j in range(len(text)):
                    char = text[j]
                    if char == "<unk>": char = "/u"
                    if char == "<eos>": char = "/e"
                    ax.text(j, i, char, ha='center', va='center', fontsize=14)

            # Add colorbar to display the color scale
            cbar = plt.colorbar(cax)

            st.pyplot(fig)

            pos = st.number_input("Select position", value=0, step=1)
            topk_ = st.slider('Select topk_:', min_value=1, max_value=100, value=10, step=1)
            plot_pos(conflict, pos, topk_)
        
        else:
            weights = prob.T[:topk, :]

            fig, ax = plt.subplots(figsize=(17, 5))

            # Create a custom colormap that transitions from white to 'viridis'
            viridis_white = plt.cm.Blues(np.linspace(0, 1, 256))
            viridis_white[:, :3] = 1 - (1 - viridis_white[:, :3]) * 0.7  # Adjust the brightness

            # Create a LinearSegmentedColormap using the custom colormap
            cmap = LinearSegmentedColormap.from_list('custom_cmap', viridis_white)

            # Set the normalization to map the weights to the range [0, 1]
            norm = Normalize(vmin=0, vmax=1)

            # Use the created colormap in imshow with the specified norm
            cax = ax.imshow(weights, cmap=cmap, interpolation='none', norm=norm)

            for i in range(token.shape[1]):
                text = tokenizer.batch_decode(token[:, i:i+1])
                for j in range(len(text)):
                    char = text[j]
                    ax.text(j, i, char, ha='center', va='center', fontsize=14)

            # Add colorbar to display the color scale
            cbar = plt.colorbar(cax)
            st.pyplot(fig)
            pos = st.number_input("Select position", value=0, step=1)
            topk_ = st.slider('Select topk_:', min_value=1, max_value=100, value=10, step=1)
            plot_pos(conflict, pos, topk_)
    else:
        print("here")

def plot_pos(conflict, pos, topk_):
    if conflict is not None:
        if conflict.shape[-1] < 500:
            categories = tkz.reverse_tokens_special(np.arange(233))
            categories = ['/p', '/s/', '/s', '/u'] + categories[4:]
            values = conflict[pos][0]

            sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
            categories = [categories[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]

            # Vẽ biểu đồ cột
            fig = plt.figure(figsize=(8, 4))
            bars = plt.bar(categories[:topk_], values[:topk_])

            # Đặt tên cho trục x và y
            plt.xlabel('Labels')
            plt.ylabel('Probs')

            plt.ylim(0, 1)
            # Hiển thị biểu đồ
            st.pyplot(fig)

@st.cache_resource()
def load_model(mtype, device):
    if mtype == 0:
        return OCRModel(False, device)
    if mtype == 1:
        return OCRModel(True, device)
    if mtype == 2:
        return TrOCR(False, device)
    if mtype == 3:
        return TrOCR(True, device)

@st.cache_resource()
def get_tkz(ttype):
    if ttype == 0:
        return AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")
    if ttype == 1:
        return TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    if ttype == 2:
        return Tokenizer()

vgg_base = load_model(0, device)
vgg_adapter = load_model(1, device)
trocr_base = load_model(2, device)
trcocr_adapter = load_model(3, device)

tokenizer = get_tkz(0)
processor = get_tkz(1)
tkz = get_tkz(2)

def main():
    st.title("Vietnamese Handwriting Text Recognition")
    img = None
    images = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if images is not None:
        for image in images:
            img = Image.open(image).convert("RGB")
            st.image(img, use_column_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        option_a = st.checkbox("Infer by Vgg Transformer")
    with col2:
        option_b = st.checkbox("Infer by Vgg Transformer with Rethinking Head")
    with col3:
        option_c = st.checkbox("Infer by TrOCR")
    with col4:
        option_d = st.checkbox("Infer by TrOCR with Rethinking Head")
    with col5:
        select_all = st.button("Select all")

    # Nếu người dùng nhấn nút "Chọn tất cả", đặt giá trị của các ô kiểm thành True
    if select_all:
        option_a = option_b = option_c = option_d = True

    # Nút "Infer" để thực hiện công việc dựa trên các tùy chọn đã chọn
    if st.button("Infer"):
        if option_a:    inference(img, 0)
        if option_b:    inference(img, 1)
        if option_c:    inference(img, 2)
        if option_d:    inference(img, 3)

    st.markdown("## Visualize probabilities")
    selected_model = st.radio('Select model:', ('VGG Transformer', 'VGG Transformer with Rethinking Head', 'TrOCR', 'TrOCR with Rethinking Head'))
    topk = st.slider('Select topk:', min_value=1, max_value=50, value=5, step=1)
    result_key = selected_model.lower().replace(" ", "_") + '_result'
    result = st.session_state.get(result_key, {})

    if result:
        plot_entire(result['token'], result['prob'], result['conflict'], topk)


    if st.button("Clear"):
        pass


if __name__ == "__main__":
    main()
