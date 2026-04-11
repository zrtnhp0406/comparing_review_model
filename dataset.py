import pandas as pd
import re

import torch
from torch.utils.data import Dataset

def clean_text(text):
    """
    Hàm làm sạch văn bản: xóa tag, xử lý xuống dòng và khoảng trắng.
    """
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_aspect_dataset(data, contain_aspect_sentences=False):
    """
    Tạo 4 dataset riêng biệt cho các khía cạnh: appearance, aroma, palate, taste.
    
    Args:
        data (pd.DataFrame): DataFrame chứa các cột sentence, beerId, profileName 
                             và các cột dự đoán aspect (0/1).
        contain_aspect_sentences (bool): Nếu True, bao gồm cả các câu trung tính dựa trên ngữ cảnh.
    """
    aspects = ['appearance_pred', 'aroma_pred', 'palate_pred', 'taste_pred']
    results_dict = {aspect: [] for aspect in aspects}

    grouped = data.groupby(['beerId', 'profileName'], sort=False)

    for (beer_id, profile_name), group in grouped:
        sentences = group.reset_index(drop=True)
        num_sentences = len(sentences)

        # Lưu trữ các câu đã lọc cho từng aspect trong review hiện tại
        current_review_aspects = {aspect: [] for aspect in aspects}

        # Biến lưu trữ aspect gần nhất được tìm thấy phía trước (dành cho logic True)
        last_seen_aspects = [] 

        for i in range(num_sentences):
            row = sentences.iloc[i]
            # Làm sạch câu trước khi xử lý
            cleaned_sentence = clean_text(row['reviewSentence'])
            
            # Nếu sau khi làm sạch mà câu trống thì bỏ qua
            if not cleaned_sentence:
                continue

            # Kiểm tra xem câu hiện tại có phải trung tính (0 0 0 0) không
            is_neutral = (row[aspects].sum() == 0)
            
            # Danh sách các aspect xuất hiện trong câu này (giá trị = 1)
            active_aspects_in_current = [a for a in aspects if row[a] == 1]

            if not is_neutral:
                # Nếu câu có aspect, thêm vào các dataset tương ứng
                for aspect in active_aspects_in_current:
                    current_review_aspects[aspect].append(cleaned_sentence)
                
                # Cập nhật aspect gần nhất cho các câu trung tính phía sau
                last_seen_aspects = active_aspects_in_current

        # Sau khi duyệt hết 1 review, nối các câu lại bằng dấu "."
        for aspect in aspects:
            if current_review_aspects[aspect]:
                # Loại bỏ dấu chấm thừa ở cuối mỗi câu con và nối lại
                processed_sentences = [s.rstrip('.') for s in current_review_aspects[aspect]]
                combined_text = ". ".join(processed_sentences) + "."
                
                results_dict[aspect].append({
                    'beerId': beer_id,
                    'profileName': profile_name,
                    'text': combined_text
                })

    # Chuyển đổi kết quả thành dictionary các DataFrame
    return {aspect: pd.DataFrame(results_dict[aspect]) for aspect in aspects}

def prepare_training_data(comparison_df, aspect_datasets_dict):
    """
    Gộp dữ liệu so sánh với dữ liệu review đã trích xuất theo aspect.
    Giữ lại toàn bộ các dòng ngay cả khi nội dung review bị thiếu (null).
    """
    all_train_samples = []
    
    # Danh sách các khía cạnh cần xử lý
    aspect_map = {
        'appearance': 'appearance_pred', 
        'aroma': 'aroma_pred',
        'palate': 'palate_pred',
        'taste': 'taste_pred'
    }

    # Lặp qua từng khía cạnh để tạo dataset riêng cho khía cạnh đó
    for label_col, aspect_key in aspect_map.items():
        aspect_df = aspect_datasets_dict[aspect_key]
        
        # Bước 1: Lấy review cho beerId_1
        merged_1 = pd.merge(
            comparison_df[['profileName', 'beerId_1', 'beerId_2', label_col]],
            aspect_df,
            left_on=['profileName', 'beerId_1'],
            right_on=['profileName', 'beerId'],
            how='left'
        ).rename(columns={'text': 'review_1'}).drop(columns=['beerId'])

        # Bước 2: Lấy review cho beerId_2
        final_merged = pd.merge(
            merged_1,
            aspect_df,
            left_on=['profileName', 'beerId_2'],
            right_on=['profileName', 'beerId'],
            how='left'
        ).rename(columns={'text': 'review_2'}).drop(columns=['beerId'])

        # Bước 3: Định dạng lại các cột theo yêu cầu
        final_merged['aspect'] = label_col
        final_merged = final_merged.rename(columns={label_col: 'aspect_pred'})
        
        # Sắp xếp lại thứ tự cột
        final_merged = final_merged[[
            'profileName', 'aspect', 'aspect_pred', 
            'beerId_1', 'review_1', 'beerId_2', 'review_2'
        ]]
        
        all_train_samples.append(final_merged)

    # --- ĐOẠN SỬA LỖI ---
    # Gộp tất cả các DataFrame trong list thành một DataFrame duy nhất
    if not all_train_samples:
        return pd.DataFrame() # Trả về DF trống nếu không có dữ liệu
        
    full_train_dataset = pd.concat(all_train_samples, ignore_index=True)

    case1 = (
        full_train_dataset['aspect_pred'].isna() &
        full_train_dataset['review_1'].notna() &
        full_train_dataset['review_2'].notna()
    )

    case2 = (
        full_train_dataset['aspect_pred'].notna() &
        (
            full_train_dataset['review_1'].isna() |
            full_train_dataset['review_2'].isna()
        )
    )

    bad_cases = case1 | case2
    num_case1 = int(case1.sum())
    num_case2 = int(case2.sum())
    print(f"Drop {bad_cases.sum()} bad samples")
    full_train_dataset = full_train_dataset[~bad_cases]

    # ===== tiếp tục pipeline cũ =====
    full_train_dataset = full_train_dataset.dropna(subset=['aspect_pred'])

    # --- LABEL MAPPING: [-1, 0, 1] → [0, 1, 2] ---
    # CrossEntropyLoss yêu cầu nhãn phải nằm trong range [0, num_classes - 1]
    # Do đó ta map: -1→0 (hòa), 0→1 (bia 1 thắng), 1→2 (bia 2 thắng)
    # Sử dụng LABEL_MAPPING dict để đảm bảo consistency
    label_mapping = {-1: 0, 0: 1, 1: 2}
    full_train_dataset['aspect_pred'] = full_train_dataset['aspect_pred'].map(label_mapping)
    
    # Kiểm tra lại lần nữa sau khi map để loại bỏ các giá trị không nằm trong mapping (nếu có)
    full_train_dataset = full_train_dataset.dropna(subset=['aspect_pred'])
    
    # Ép kiểu sang int để đưa vào CrossEntropyLoss
    full_train_dataset['aspect_pred'] = full_train_dataset['aspect_pred'].astype(int)

    return full_train_dataset,num_case1, num_case2

class BeerComparisonDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True) # Reset index để tránh lỗi idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
    
        review_1 = str(row['review_1'])
        review_2 = str(row['review_2'])
        
        # Tokenize review 1
        encoding_1 = self.tokenizer(
            review_1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize review 2
        encoding_2 = self.tokenizer(
            review_2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
            # Lấy nhãn (aspect_pred)
        label = torch.tensor(row['aspect_pred'], dtype=torch.long)

        return {
            'input_ids_1': encoding_1['input_ids'].flatten(),
            'mask_1': encoding_1['attention_mask'].flatten(),
            'input_ids_2': encoding_2['input_ids'].flatten(),
            'mask_2': encoding_2['attention_mask'].flatten(),
            'labels': label
        }


# ============================================
# LABEL MAPPING REFERENCE
# ============================================
# Mapping sử dụng trong prepare_training_data():
# Nhãn gốc [-1, 0, 1] được chuyển thành [0, 1, 2] để phù hợp với CrossEntropyLoss
LABEL_MAPPING = {
    -1: 0,  # Hòa (draw)
    0: 1,   # Bia 1 thắng (beer 1 wins)
    1: 2    # Bia 2 thắng (beer 2 wins)
}

# Mapping ngược để decode dự đoán
REVERSE_LABEL_MAPPING = {
    0: -1,  # Output 0 → Nhãn gốc -1 (hòa)
    1: 0,   # Output 1 → Nhãn gốc 0 (bia 1 thắng)
    2: 1    # Output 2 → Nhãn gốc 1 (bia 2 thắng)
}


def decode_label(pred_label):
    """
    Chuyển đổi nhãn dự đoán từ [0, 1, 2] về lại nhãn gốc [-1, 0, 1]
    
    Args:
        pred_label: Nhãn dự đoán từ model (0, 1, hoặc 2)
        
    Returns:
        Nhãn gốc (-1, 0, hoặc 1)
    """
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.item()
    return REVERSE_LABEL_MAPPING.get(int(pred_label), None)
