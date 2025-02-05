# %%
import matplotlib.pyplot as plt
import pandas as pd

# Hàm vẽ đồ thị Loss
def plot_loss(num_epochs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, eval_losses, label="Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.show()
# Hàm vẽ đồ thị BLEU Scores
def plot_bleu(num_epochs, train_bleu_scores, eval_bleu_scores):
    epochs = list(range(num_epochs))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_bleu_scores, label="Training BLEU Score", linestyle="--")
    plt.plot(epochs, eval_bleu_scores, label="Evaluation BLEU Score", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Scores over Epochs")
    plt.legend()
    plt.show()

# Đọc dữ liệu CSV
df = pd.read_csv(
    filepath_or_buffer=r"Scores.csv", 
    skiprows=1,    # Bỏ qua header
    header=None,   # Không đọc dòng đầu làm header
    delimiter=','  # Dấu phẩy để tách cột
)
num_epochs = df.shape[0]
# Gọi hàm vẽ Loss
plot_loss(
    num_epochs=num_epochs, 
    train_losses=df[0], 
    eval_losses=df[1]
)
# Gọi hàm vẽ BLEU Scores
plot_bleu(
    num_epochs=num_epochs, 
    train_bleu_scores=df[2], 
    eval_bleu_scores=df[3]
)