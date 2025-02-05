import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from data_processing import src_vocab, tgt_vocab
from model_architecture import RnnDecoder, RnnEncoder, RnnMachineTranslate
from model_architecture import CONTRACTIONS, TextProcessor, OutputProcessor

# Hàm vẽ đồ thị Loss
def plot_loss(num_epochs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Training Loss")
    ax.plot(epochs, eval_losses, label="Evaluation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Epochs")
    ax.legend()
    return fig

# Hàm vẽ đồ thị BLEU Scores
def plot_bleu(num_epochs, train_bleu_scores, eval_bleu_scores):
    epochs = list(range(num_epochs))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_bleu_scores, label="Training BLEU Score", linestyle="--")
    ax.plot(epochs, eval_bleu_scores, label="Evaluation BLEU Score", linestyle="--")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("BLEU Score")
    ax.set_title("BLEU Scores over Epochs")
    ax.legend()
    return fig

# Đọc dữ liệu từ CSV
df = pd.read_csv(
    filepath_or_buffer=r"Scores.csv", 
    skiprows=1,    # Bỏ qua header
    header=None,   # Không đọc dòng đầu làm header
    delimiter=','  # Dấu phẩy để tách cột
)
num_epochs = df.shape[0]

# Streamlit app setup
st.set_page_config(page_title="Machine Translation")
st.title("Machine Translation Training and Translation")

# Thêm phần hiển thị đồ thị huấn luyện
st.subheader("Training Visualization")

# Vẽ và hiển thị đồ thị Loss
st.write("### Loss over Epochs")
loss_fig = plot_loss(num_epochs, df[0], df[1])
st.pyplot(loss_fig)

# Vẽ và hiển thị đồ thị BLEU Scores
st.write("### BLEU Scores over Epochs")
bleu_fig = plot_bleu(num_epochs, df[2], df[3])
st.pyplot(bleu_fig)

# Phần dịch máy
st.write("## Translate text from English to Vietnamese")

embed_dim = 512
hidden_dim = 1024
n_layers = 6
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo các thành phần của mô hình
processor = TextProcessor(CONTRACTIONS, src_vocab, DEVICE)
out_processor = OutputProcessor(tgt_vocab, DEVICE)
encoder = RnnEncoder(src_vocab, embed_dim, hidden_dim, n_layers, 0.2, DEVICE).to(DEVICE)
decoder = RnnDecoder(tgt_vocab, hidden_dim, n_layers, 0.2, SOS_IDX, EOS_IDX, PAD_IDX, DEVICE, 0).to(DEVICE)
translation_model = RnnMachineTranslate(encoder, decoder).to(DEVICE)

# Load mô hình đã huấn luyện
translation_model.load_state_dict(torch.load(r"machine_translation.pth", weights_only=True))
translation_model.eval()

# Tạo input text area
input_text = st.text_area("Enter English text:", height=100)

# Nút "Translate"
if st.button("Translate"):
    sen = processor.process(input_text)
    output = translation_model.inference(sen)
    output = out_processor.process(output)
    # Hiển thị kết quả trong khung
    st.subheader("Translated Text:")
    st.text_area(
        label="Translated text (disabled)",
        value=output,
        height=100,
        disabled=True,
        label_visibility="collapsed"
    )

