import math
from xml.parsers.expat import model
import torch
import torch.nn as nn
from tqdm import tqdm # Thư viện để hiển thị thanh progress bar
from torch.utils.data import DataLoader
from dataset import BeerComparisonDataset
from transformers import get_linear_schedule_with_warmup
from utilize import compute_metrics, calculate_class_metrics, calculate_summary_metrics, print_class_report
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import numpy as np
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, train_df, val_df, test_df, fp_train, fn_train, fp_test, fn_test, fp_val, fn_val, tokenizer, num_epochs=15, batch_size=4, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    set_seed(42)
    # 1. Khởi tạo Dataset & DataLoader
    train_dataset = BeerComparisonDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = BeerComparisonDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = BeerComparisonDataset(test_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 2. Thiết lập Optimizer với Weight Decay (0.01)
    # Loại bỏ weight decay cho các tham số bias và LayerNorm
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-4)

    # 3. Thiết lập Gradient Accumulation & Scheduler
    accumulation_steps = 4
    # Tổng số bước update thực tế = (tổng batch / accumulation) * số epoch
    total_steps = math.ceil(len(train_loader) / accumulation_steps) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
        num_training_steps=total_steps
    )

    weight = torch.tensor([1.0, 1.0, 1.0]).to(device)  # đánh mạnh vào lớp 0
    criterion = nn.CrossEntropyLoss(weight=weight)

    # 4. Biến cho Early Stopping
    best_val_f1 = 0
    patience_counter = 0
    patience = 5

    print(f"Bắt đầu huấn luyện trên: {device}")
    
    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for step, batch in enumerate(train_loop):
            input_ids_1 = batch['input_ids_1'].to(device)
            mask_1 = batch['mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            mask_2 = batch['mask_2'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids_1, mask_1, input_ids_2, mask_2)
            
            loss = criterion(logits, labels)
            
            # Chia loss cho accumulation_steps
            loss = loss / accumulation_steps
            loss.backward()

            # Chỉ update trọng số sau khi tích lũy đủ steps
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps
            train_loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_loop:
                input_ids_1 = batch['input_ids_1'].to(device)
                mask_1 = batch['mask_1'].to(device)
                input_ids_2 = batch['input_ids_2'].to(device)
                mask_2 = batch['mask_2'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids_1, mask_1, input_ids_2, mask_2)

                _, preds = torch.max(logits, dim=1)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Tính toán Metrics bằng các hàm utilize của bạn
        tp, fp, fn = compute_metrics(all_labels, all_preds, fp_val, fn_val)
        class_metrics = calculate_class_metrics(tp, fp, fn)
        summary_metrics = calculate_summary_metrics(tp, fp, fn, class_metrics)
        
        current_val_f1 = summary_metrics['macro_f1']
        avg_val_loss = total_val_loss / len(val_loader)
            
        print(f"\n===== Validation Metrics (Epoch {epoch+1}) =====")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Macro P/R/F1: {summary_metrics['macro_precision']:.4f} / {summary_metrics['macro_recall']:.4f} / {current_val_f1:.4f}")
        print_class_report(class_metrics)

        # --- LOGIC EARLY STOPPING (Theo Macro F1) ---
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_esim_bert.pt")
            print(f"--> Đã lưu model tốt nhất với Macro F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"--> EarlyStopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Dừng sớm do hiệu suất không cải thiện!")
                break
    model.load_state_dict(torch.load("best_esim_bert.pt"))
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0
    
    test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
    
    with torch.no_grad():
        for batch in test_loop:
            input_ids_1 = batch['input_ids_1'].to(device)
            mask_1 = batch['mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            mask_2 = batch['mask_2'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids_1, mask_1, input_ids_2, mask_2)

            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính toán Metrics bằng các hàm utilize của bạn
    tp, fp, fn = compute_metrics(all_labels, all_preds, fp_test, fn_test)
    class_metrics = calculate_class_metrics(tp, fp, fn)
    summary_metrics = calculate_summary_metrics(tp, fp, fn, class_metrics)
    current_val_f1 = summary_metrics['macro_f1']
    print(f"\n===== Testing Metrics =====")
    print(f"Macro P/R/F1: {summary_metrics['macro_precision']:.4f} / {summary_metrics['macro_recall']:.4f} / {current_val_f1:.4f}")
    print_class_report(class_metrics)
    return model
