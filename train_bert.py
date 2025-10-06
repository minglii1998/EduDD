import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import argparse
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 自定义数据集类
class DifficultyDataset(Dataset):
    def __init__(self, texts, difficulties, tokenizer, max_length=512):
        self.texts = texts
        self.difficulties = difficulties
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        difficulty = self.difficulties[idx]
        
        # 使用tokenizer编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'difficulty': torch.tensor(difficulty, dtype=torch.float)
        }

# 2. 加载数据函数
def load_data(file_path, training_target):
    """从JSON文件加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['processed_text'] for item in data]
    difficulties = [item[training_target] for item in data]
    
    return texts, difficulties

# 3. 定义Transformer回归模型（支持BERT/Longformer等）
class TransformerRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        super(TransformerRegressor, self).__init__()
        self.model_name = model_name
        
        cache_directory = os.path.expanduser('/nfshomes/minglii/scratch/cache/hub')
        
        try:
            self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_directory)
        except OSError:
            print(f"PyTorch weights not found, loading from TensorFlow weights...")
            self.transformer = AutoModel.from_pretrained(model_name, from_tf=True, cache_dir=cache_directory)
            
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.transformer.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        #     pooled_output = outputs.pooler_output
        # else:
        #     pooled_output = outputs.last_hidden_state[:, 0, :]

        last_hidden = outputs.last_hidden_state                        # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()                    # [B, L, 1]
        summed = (last_hidden * mask).sum(dim=1)                       # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
        pooled_output = summed / denom                                 # [B, H]
        
        output = self.dropout(pooled_output)
        output = self.regressor(output)
        
        return output.squeeze()

# 4. 训练函数
def train_epoch(model, dataloader, optimizer, device, criterion, scheduler=None):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        difficulties = batch['difficulty'].to(device)
        
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, difficulties)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新learning rate scheduler（每个step）
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 5. 评估函数
def eval_model(model, dataloader, device, criterion, scaler=None):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        criterion: 损失函数
        scaler: StandardScaler对象，如果提供则将预测值和真实值转换回原始scale
    """
    model.eval()
    predictions_list = []
    actuals_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulties = batch['difficulty'].to(device)
            
            predictions = model(input_ids, attention_mask)
            
            # --- FIX: Ensure predictions are always iterable ---
            predictions_np = predictions.cpu().numpy()
            predictions_list.extend(np.atleast_1d(predictions_np))
            actuals_list.extend(np.atleast_1d(difficulties.cpu().numpy()))
    
    predictions_array = np.array(predictions_list)
    actuals_array = np.array(actuals_list)
    
    # 如果提供了scaler，转换回原始scale
    if scaler is not None:
        predictions_array = scaler.inverse_transform(predictions_array.reshape(-1, 1)).flatten()
        actuals_array = scaler.inverse_transform(actuals_array.reshape(-1, 1)).flatten()
    
    # 计算所有指标（在原始scale上）
    mse = np.mean((predictions_array - actuals_array) ** 2)
    mae = np.mean(np.abs(predictions_array - actuals_array))
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_array, predictions_array)
    
    # 返回原始scale的值（如果使用了scaler则已转换，否则保持原样）
    return mse, mae, rmse, r2, predictions_array.tolist(), actuals_array.tolist()

# 6. 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='BERT Difficulty Regression Training')
    
    # 数据相关参数
    parser.add_argument('--train_file', type=str, default='Cambridge_train.json', 
                        help='Path to training data file')
    parser.add_argument('--val_file', type=str, default=None, 
                        help='Path to validation data file (optional, if not provided, will split from train_file)')
    parser.add_argument('--test_file', type=str, default='Cambridge_test.json', 
                        help='Path to test data file')
    parser.add_argument('--val_split', type=float, default=0.15, 
                        help='Validation split ratio (0-1), used only if val_file is not provided (default: 0.15)')
    parser.add_argument('--training_target', type=str, default='difficulty', 
                        help='difficulty, discrimination, facility')

    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', 
                        help='Model name (e.g., bert-base-uncased, allenai/longformer-base-4096)')
    parser.add_argument('--max_length', type=int, default=512, 
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.3, 
                        help='Dropout rate')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, 
                        help='Ratio of total steps for learning rate warmup (default: 0.1)')
    parser.add_argument('--warmup_steps', type=int, default=None, 
                        help='Number of warmup steps (overrides warmup_ratio if set)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save model and results')
    parser.add_argument('--save_model', type=str, default='best_model.pt', 
                        help='Filename for saving the best model')
    parser.add_argument('--normalize', action='store_true', 
                        help='Apply z-score normalization to difficulty values')
    
    return parser.parse_args()

# 7. 设置随机种子
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 8. 主函数
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 50)
    
    # ========== 加载数据 ==========
    print("\nLoading data...")
    
    # 加载训练数据
    train_texts, train_difficulties = load_data(args.train_file, args.training_target)
    print(f"Loaded training data: {len(train_texts)} samples")
    
    # 加载或分割验证数据
    if args.val_file is not None:
        # 如果提供了验证集文件，直接加载
        val_texts, val_difficulties = load_data(args.val_file, args.training_target)
        print(f"Loaded validation data: {len(val_texts)} samples")
    else:
        # 如果没有提供验证集，从训练集分割
        print(f"No validation file provided, splitting {args.val_split*100:.0f}% from training data...")
        train_texts, val_texts, train_difficulties, val_difficulties = train_test_split(
            train_texts, train_difficulties, 
            test_size=args.val_split, 
            random_state=args.seed,
            shuffle=True
        )
        print(f"Split result - Train: {len(train_texts)} samples, Val: {len(val_texts)} samples")
    
    # 加载测试数据
    test_texts, test_difficulties = load_data(args.test_file)
    print(f"Loaded test data: {len(test_texts)} samples")
    print("-" * 50)
    
    # ========== Z-score标准化 ==========
    scaler = None
    if args.normalize:
        print("\nApplying z-score normalization to difficulty values...")
        scaler = StandardScaler()
        
        # 只在训练集上fit scaler
        train_difficulties_array = np.array(train_difficulties).reshape(-1, 1)
        scaler.fit(train_difficulties_array)
        
        # 转换所有数据集
        train_difficulties = scaler.transform(train_difficulties_array).flatten().tolist()
        val_difficulties = scaler.transform(np.array(val_difficulties).reshape(-1, 1)).flatten().tolist()
        test_difficulties = scaler.transform(np.array(test_difficulties).reshape(-1, 1)).flatten().tolist()
        
        print(f"  Training data - Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
        print(f"  Normalized training range: [{min(train_difficulties):.4f}, {max(train_difficulties):.4f}]")
        print("-" * 50)
    else:
        print("\nSkipping normalization (--normalize flag not set)")
        print("-" * 50)
    
    # ========== 初始化tokenizer ==========
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = DifficultyDataset(train_texts, train_difficulties, tokenizer, args.max_length)
    val_dataset = DifficultyDataset(val_texts, val_difficulties, tokenizer, args.max_length)
    test_dataset = DifficultyDataset(test_texts, test_difficulties, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = TransformerRegressor(model_name=args.model_name, dropout=args.dropout).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # ========== 创建learning rate scheduler ==========
    # 计算总训练步数
    total_steps = len(train_loader) * args.epochs
    
    # 计算warmup步数
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 创建linear warmup + decay scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nLearning rate scheduler:")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"  Initial LR: {args.lr}")
    print("-" * 50)
    
    best_val_loss = float('inf')
    training_history = []
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, scheduler=scheduler)
        # 评估时传入scaler，自动转换回原始scale计算指标
        val_mse, val_mae, val_rmse, val_r2, _, _ = eval_model(model, val_loader, device, criterion, scaler=scaler)
        
        # 获取当前learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'  Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}')
        if args.normalize:
            print(f'  Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f} (original scale)')
        else:
            print(f'  Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}')
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_mse': float(val_mse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_r2': float(val_r2)
        })
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            model_path = os.path.join(args.output_dir, args.save_model)
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ Best model saved to {model_path}!')
        print('-' * 50)
    
    print('\nEvaluating on test set...')
    model_path = os.path.join(args.output_dir, args.save_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 测试时也传入scaler，转换回原始scale
    test_mse, test_mae, test_rmse, test_r2, test_predictions, test_actuals = eval_model(
        model, test_loader, device, criterion, scaler=scaler
    )
    
    if args.normalize:
        print(f'Test MSE: {test_mse:.4f} (original scale)')
        print(f'Test MAE: {test_mae:.4f} (original scale)')
        print(f'Test RMSE: {test_rmse:.4f} (original scale)')
        print(f'Test R²: {test_r2:.4f}')
    else:
        print(f'Test MSE: {test_mse:.4f}')
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')
        print(f'Test R²: {test_r2:.4f}')
    
    results = {
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'best_val_loss': float(best_val_loss),
        'normalized': args.normalize,
        'training_history': training_history,
        'predictions': [float(p) for p in test_predictions],
        'actuals': [float(a) for a in test_actuals]
    }
    
    # 如果使用了标准化，保存scaler信息
    if args.normalize:
        results['scaler_mean'] = float(scaler.mean_[0])
        results['scaler_std'] = float(scaler.scale_[0])
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nTest results saved to {results_path}')
    
    print('\nSample predictions on test set:')
    for i in range(min(5, len(test_predictions))):
        print(f'  Sample {i+1}: Predicted={test_predictions[i]:.4f}, Actual={test_actuals[i]:.4f}')
    
    print('\nTraining completed!')

if __name__ == '__main__':
    main()