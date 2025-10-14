import json
import sys
import datetime
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

# 日志 Tee 实现：将输出同时写到控制台和文件
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_LOG_FH = None

# 1. 自定义数据集类 (重构以支持多输入)
class MultiInputDataset(Dataset):
    def __init__(self, questions, answers_list, difficulties, tokenizer, max_length=512):
        self.questions = questions
        self.answers_list = answers_list
        self.difficulties = difficulties
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert len(self.questions) == len(self.answers_list) == len(self.difficulties)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers_list[idx]
        difficulty = self.difficulties[idx]
        all_texts = [question] + answers
        all_input_ids = []
        all_attention_masks = []
        for text in all_texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            all_input_ids.append(encoding['input_ids'])
            all_attention_masks.append(encoding['attention_mask'])
        
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'difficulty': torch.tensor(difficulty, dtype=torch.float)
        }

# 2. 加载数据函数 (更新以处理多个答案)
def load_data(file_path, training_target, answer_keys=None):
    if answer_keys is None:
        answer_keys = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item['processed_text'] for item in data]
    difficulties = [item[training_target] for item in data]
    answers_lists = []
    
    for item in data:
        current_answers = [item[key] for key in answer_keys if key in item]
        answers_lists.append(current_answers)
    
    return questions, answers_lists, difficulties

# 3. 定义新的多输入Transformer回归模型
class MultiInputTransformerRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3, cache_dir=None, nhead=8):
        super(MultiInputTransformerRegressor, self).__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = self.transformer.config.hidden_size
        self.self_attention_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask):
        B, N, L = input_ids.shape
        H = self.transformer.config.hidden_size
        input_ids_flat = input_ids.view(B * N, L)
        attention_mask_flat = attention_mask.view(B * N, L)
        
        outputs = self.transformer(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        
        last_hidden = outputs.last_hidden_state
        mask = attention_mask_flat.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled_output_flat = summed / denom
        
        all_embeddings = pooled_output_flat.view(B, N, H)
        
        q_embeddings = all_embeddings[:, 0, :]
        a_embeddings = all_embeddings[:, 1:, :]
        
        if a_embeddings.shape[1] > 0:
            a_attention_output = self.self_attention_layer(a_embeddings)
            agg_a_embeddings = a_attention_output.mean(dim=1)
        else:
            agg_a_embeddings = torch.zeros_like(q_embeddings)

        combined_features = torch.cat([q_embeddings, agg_a_embeddings], dim=1)
        
        output = self.dropout(combined_features)
        output = self.regressor(output)
        
        return output.squeeze(-1)

# 4. 训练函数 (添加梯度累积)
def train_epoch(model, dataloader, optimizer, device, criterion, scheduler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        difficulties = batch['difficulty'].to(device)
        
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, difficulties)
        
        # 将loss除以累积步数，使得累积后的梯度与原始batch size相当
        loss = loss / accumulation_steps
        loss.backward()
        
        # 每accumulation_steps步或最后一个batch时更新参数
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
        
        # 记录未缩放的loss用于监控
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)

# 5. 评估函数
def eval_model(model, dataloader, device, criterion, scaler=None):
    model.eval()
    predictions_list = []
    actuals_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulties = batch['difficulty'].to(device)
            
            predictions = model(input_ids, attention_mask)
            
            predictions_np = predictions.cpu().numpy()
            predictions_list.extend(np.atleast_1d(predictions_np))
            actuals_list.extend(np.atleast_1d(difficulties.cpu().numpy()))
    
    predictions_array = np.array(predictions_list)
    actuals_array = np.array(actuals_list)
    
    if scaler is not None:
        predictions_array = scaler.inverse_transform(predictions_array.reshape(-1, 1)).flatten()
        actuals_array = scaler.inverse_transform(actuals_array.reshape(-1, 1)).flatten()
    
    mse = np.mean((predictions_array - actuals_array) ** 2)
    mae = np.mean(np.abs(predictions_array - actuals_array))
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_array, predictions_array)
    
    return mse, mae, rmse, r2, predictions_array.tolist(), actuals_array.tolist()

# 6. 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Input Transformer Difficulty Regression')
    parser.add_argument('--train_file', type=str, default='Cambridge_train.json')
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default='Cambridge_test.json')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--training_target', type=str, default='difficulty')
    parser.add_argument('--answer_keys', type=str, nargs='+', default=[], help='A list of keys for the answers to include from the JSON file (e.g., --answer_keys answer1 answer2)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attention_heads', type=int, default=8, help='Number of heads in the self-attention layer for answers')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps. Effective batch size = batch_size * accumulation_steps')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./output_multi_input')
    parser.add_argument('--save_model', type=str, default='best_model_multi.pt')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--cache_dir', type=str, default='/nfshomes/minglii/scratch/cache/hub')
    parser.add_argument('--log_file', type=str, default=None)
    
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

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    default_log = os.path.join(args.output_dir, f'train_multi_{timestamp}.log')
    log_path = args.log_file if args.log_file is not None else default_log
    
    if os.path.exists(log_path):
        os.remove(log_path)
    
    global _LOG_FH
    _LOG_FH = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, _LOG_FH)
    sys.stderr = Tee(sys.stderr, _LOG_FH)
    print(f"Logging to {log_path}")
    
    print("=" * 50)
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 50)
    
    # 计算有效batch size
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"\nEffective batch size: {args.batch_size} × {args.accumulation_steps} = {effective_batch_size}")
    
    print("\nLoading data...")
    train_questions, train_answers, train_difficulties = load_data(args.train_file, args.training_target, args.answer_keys)
    print(f"Loaded training data: {len(train_questions)} samples, with {len(args.answer_keys)} answers per question.")
    
    if args.val_file is not None:
        val_questions, val_answers, val_difficulties = load_data(args.val_file, args.training_target, args.answer_keys)
        print(f"Loaded validation data: {len(val_questions)} samples.")
    else:
        print(f"Splitting validation set from training data...")
        train_questions, val_questions, train_answers, val_answers, train_difficulties, val_difficulties = train_test_split(
            train_questions, train_answers, train_difficulties,
            test_size=args.val_split, 
            random_state=args.seed,
            shuffle=True
        )
        print(f"Split result - Train: {len(train_questions)} samples, Val: {len(val_questions)} samples")
    
    test_questions, test_answers, test_difficulties = load_data(args.test_file, args.training_target, args.answer_keys)
    print(f"Loaded test data: {len(test_questions)} samples.")
    print("-" * 50)

    scaler = None
    if args.normalize:
        print("\nApplying z-score normalization...")
        scaler = StandardScaler()
        train_difficulties_array = np.array(train_difficulties).reshape(-1, 1)
        scaler.fit(train_difficulties_array)
        train_difficulties = scaler.transform(train_difficulties_array).flatten().tolist()
        val_difficulties = scaler.transform(np.array(val_difficulties).reshape(-1, 1)).flatten().tolist()
        test_difficulties = scaler.transform(np.array(test_difficulties).reshape(-1, 1)).flatten().tolist()
        print(f"  Normalization applied. Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")
    
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    train_dataset = MultiInputDataset(train_questions, train_answers, train_difficulties, tokenizer, args.max_length)
    val_dataset = MultiInputDataset(val_questions, val_answers, val_difficulties, tokenizer, args.max_length)
    test_dataset = MultiInputDataset(test_questions, test_answers, test_difficulties, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MultiInputTransformerRegressor(
        model_name=args.model_name, 
        dropout=args.dropout, 
        cache_dir=args.cache_dir,
        nhead=args.attention_heads
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # 计算总步数时考虑梯度累积
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_steps is None else args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print(f"\nScheduler: {total_steps} total optimization steps, {warmup_steps} warmup steps.")
    print(f"Note: Total steps adjusted for gradient accumulation (original batches: {len(train_loader) * args.epochs})")
    print("-" * 50)
    
    best_val_loss = float('inf')
    training_history = []
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, scheduler, args.accumulation_steps)
        val_mse, val_mae, val_rmse, val_r2, _, _ = eval_model(model, val_loader, device, criterion, scaler=scaler)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | LR: {current_lr:.2e}')
        
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
    
    print('\nEvaluating on test set with the best model...')
    model_path = os.path.join(args.output_dir, args.save_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_mse, test_mae, test_rmse, test_r2, test_predictions, test_actuals = eval_model(model, test_loader, device, criterion, scaler=scaler)
    
    print(f'Test Results (original scale):')
    print(f'  MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}')
    
    results = {
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'best_val_loss': float(best_val_loss),
        'normalized': args.normalize,
        'effective_batch_size': effective_batch_size,
        'accumulation_steps': args.accumulation_steps,
        'training_history': training_history,
        'predictions': [float(p) for p in test_predictions],
        'actuals': [float(a) for a in test_actuals]
    }
    if args.normalize:
        results['scaler_mean'] = float(scaler.mean_[0])
        results['scaler_std'] = float(scaler.scale_[0])
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nTest results saved to {results_path}')
    
    print('\nTraining completed!')

if __name__ == '__main__':
    main()