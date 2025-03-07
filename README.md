# Dịch máy với mô hình mạng hồi quy
Dự án này triển khai một hệ thống dịch máy dựa trên Mạng Nơ-ron Hồi quy (RNN) sử dụng PyTorch và TorchText. Hệ thống dịch văn bản từ tiếng Anh sang tiếng Việt bằng cách sử dụng kiến trúc mã hóa-giải mã với các tầng GRU (Gated Recurrent Unit). Dự án bao gồm tiền xử lý dữ liệu, huấn luyện mô hình và giao diện Streamlit để trực quan hóa và suy luận.

## Mục lục
1. [Tổng Quan Dự Án](#tổng-quan-dự-án)
2. [Tính Năng](#tính-năng)
3. [Yêu Cầu](#yêu-cầu)
4. [Cài Đặt](#cài-đặt)
5. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
   - [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
   - [Chạy Ứng Dụng Streamlit](#chạy-ứng-dụng-streamlit)
6. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
7. [Chi Tiết Kỹ Thuật](#chi-tiết-kỹ-thuật)
   - [Tiền Xử Lý Dữ Liệu](#tiền-xử-lý-dữ-liệu)
   - [Kiến Trúc Mô Hình](#kiến-trúc-mô-hình)
   - [Huấn Luyện và Đánh Giá](#huấn-luyện-và-đánh-giá)
8. [Giấy Phép](#giấy-phép)

## Tổng quan dự án 
Dự án này xây dựng một mô hình dịch máy để dịch các câu tiếng Anh sang tiếng Việt. Nó sử dụng GRU hai chiều cho mã hóa và GRU với cơ chế teacher forcing cho giải mã. Hệ thống tiền xử lý dữ liệu văn bản (ví dụ: chuẩn hóa các từ viết tắt và định dạng thời gian), vector hóa dữ liệu bằng từ điển, và huấn luyện mô hình với DataLoader tùy chỉnh. Mô hình đã huấn luyện được triển khai thông qua ứng dụng Streamlit để dịch văn bản theo thời gian thực và trực quan hóa các chỉ số huấn luyện (Loss và điểm BLEU).

Mã nguồn được thiết kế theo kiểu mô-đun, với các file riêng biệt cho tiền xử lý dữ liệu, kiến trúc mô hình và giao diện Streamlit.

## Tính năng
- Chuẩn Hóa Văn Bản: Xử lý các từ viết tắt (ví dụ: "can't" → "can not") và định dạng thời gian (ví dụ: "2:30" → "2 giờ 30 phút").
Xây Dựng Từ Điển: Tạo từ điển cho ngôn ngữ nguồn (tiếng Anh) và ngôn ngữ đích (tiếng Việt) với các token đặc biệt (<unk>, <pad>, <sos>, <eos>).
- Kiến Trúc RNN: Sử dụng GRU hai chiều cho mã hóa và GRU với teacher forcing cho giải mã.
- DataLoader: Dataset tùy chỉnh và đệm cho các chuỗi có độ dài thay đổi.
- Giao Diện Streamlit: Trực quan hóa Loss và điểm BLEU trong quá trình huấn luyện, cung cấp giao diện dịch văn bản.
- Suy Luận: Tạo bản dịch cho các đầu vào tiếng Anh mới.

## Yêu cầu
- Python 3.8 trở lên
- PyTorch
- TorchText
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- CUDA (tùy chọn, để hỗ trợ GPU)

## Hướng dẫn sử dụng
__Huấn luyện mô hình__

1. Chuẩn bị tập dữ liệu trong data.csv với hai cột: văn bản nguồn (tiếng Anh) và văn bản đích (tiếng Việt), cách nhau bằng dấu tab.
2. Chạy script tiền xử lý dữ liệu và huấn luyện:
    
        python data_processing.py
        python model_architecture.py
    (Lưu ý: Logic huấn luyện chưa được cung cấp đầy đủ trong tài liệu; bạn có thể cần triển khai dựa trên train_loader, val_loader, và test_loader.)
3. Lưu mô hình đã huấn luyện dưới dạng machine_translation.pth.

__Chạy ứng dụng Streamlit__

1. Đảm bảo Scores.csv chứa các chỉ số huấn luyện (các cột: train_loss, eval_loss, train_bleu, eval_bleu).
2. Khởi chạy ứng dụng Streamlit:
        
        streamlit run streamlit_app.py
3. Mở trình duyệt tại http://localhost:8501 để xem trực quan hóa huấn luyện và dịch văn bản.

## Cấu trúc thư mục

    machine-translation-rnn/
    │
    ├── data.csv              # Tập dữ liệu chứa cặp câu tiếng Anh-tiếng Việt
    ├── Scores.csv            # Chỉ số huấn luyện (Loss và điểm BLEU)
    ├── machine_translation.pth # Trọng số mô hình đã huấn luyện
    ├── data_processing.py    # Tiền xử lý dữ liệu và thiết lập DataLoader
    ├── model_architecture.py # Kiến trúc mô hình RNN mã hóa-giải mã và bộ xử lý văn bản
    ├── streamlit_app.py      # Giao diện Streamlit cho trực quan hóa và suy luận
    ├── README.md             # Tài liệu dự án
    └── requirements.txt      # Danh sách thư viện Python cần thiết

## Chi Tiết Kỹ Thuật
__Tiền Xử Lý Dữ Liệu__

1. Chuẩn Hóa: Mở rộng các từ viết tắt, chuẩn hóa dấu câu và chuyển đổi định dạng thời gian.
2. Tokenization: Sử dụng tokenizer basic_english từ TorchText.
3. Từ Điển: Xây dựng từ điển cho ngôn ngữ nguồn và đích với tối đa 200,000 token mỗi loại.
4. Vector Hóa: Thêm các token đặc biệt (<sos>, <eos>) và đệm các chuỗi cho xử lý theo batch.

__Kiến Trúc Mô Hình__
1. Mã Hóa: GRU hai chiều với số tầng, kích thước embedding và kích thước ẩn có thể cấu hình. Đầu ra bao gồm các bước thời gian và trạng thái ẩn cuối cùng.
2. Giải Mã: GRU với teacher forcing (tỷ lệ có thể cấu hình, mặc định 0.3). Tạo chuỗi đích từng bước.
3. Mô Hình Dịch: Kết hợp mã hóa và giải mã, hỗ trợ cả chế độ huấn luyện và suy luận.
4. Bộ Xử Lý Văn Bản: Chuẩn hóa và biến đổi văn bản đầu vào thành tensor tương thích với mô hình.
5. Bộ Xử Lý Đầu Ra: Chuyển đổi đầu ra mô hình (chỉ số token) thành câu có thể đọc được.

__Huấn Luyện và Đánh Giá__
1. Chia Tập Dữ Liệu: 70% huấn luyện, 20% xác thực, 10% kiểm tra (sau khi điều chỉnh).
2. Kích Thước Batch: 32 cho huấn luyện, 8 cho xác thực/kiểm tra.
3. Chỉ Số: Theo dõi và trực quan hóa Loss (ví dụ: cross-entropy) và điểm BLEU.

## Giấy Phép
Dự án này được cấp phép theo Giấy phép MIT. Xem chi tiết trong file LICENSE.
