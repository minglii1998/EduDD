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

# 1. 自定义数据集类 (Data Augmentation版本)
class AugmentedDataset(Dataset):
    def __init__(self, questions, answers_list, difficulties, tokenizer, max_length=512, question_ids=None):
        """
        将数据展开：每个(question, answer)对作为一个样本
        question_ids用于在评估时识别哪些样本属于同一个问题
        """
        self.samples = []
        self.question_ids = []
        
        for idx, (question, answers, difficulty) in enumerate(zip(questions, answers_list, difficulties)):
            if len(answers) == 0:
                # 如果没有答案，只用问题本身
                self.samples.append({
                    'text': question,
                    'difficulty': difficulty
                })
                self.question_ids.append(idx)
            else:
                # 每个答案创建一个样本
                for answer in answers:
                    combined_text = question + " [SEP] " + answer
                    self.samples.append({
                        'text': combined_text,
                        'difficulty': difficulty
                    })
                    self.question_ids.append(idx)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        difficulty = sample['difficulty']
        
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'difficulty': torch.tensor(difficulty, dtype=torch.float),
            'question_id': self.question_ids[idx]
        }

# 2. 加载数据函数
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

# 3. 简化的Transformer回归模型
class SimpleTransformerRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.3, cache_dir=None):
        super(SimpleTransformerRegressor, self).__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled_output = summed / denom
        
        output = self.dropout(pooled_output)
        output = self.regressor(output)
        
        return output.squeeze(-1)

# 4. 训练函数 (带gradient accumulation)
def train_model(model, train_loader, val_loader, optimizer, device, criterion, scheduler, 
                args, scaler=None):
    """
    训练主循环，支持gradient accumulation和按step进行评估
    """
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    training_history = []
    accumulated_loss = 0.0
    optimizer.zero_grad()
    
    print("\nStarting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Evaluation every {args.eval_steps} steps")
    print("-" * 50)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulties = batch['difficulty'].to(device)
            
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, difficulties)
            
            # 归一化loss（除以accumulation steps）
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            accumulated_loss += loss.item()
            epoch_loss += loss.item()
            
            # 每accumulation_steps步或最后一个batch更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 打印训练进度
                if global_step % args.log_steps == 0:
                    avg_loss = accumulated_loss / args.log_steps
                    current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
                    print(f'Epoch {epoch+1}/{args.epochs} | Step {global_step} | '
                          f'Loss: {avg_loss:.4f} | LR: {current_lr:.2e}')
                    accumulated_loss = 0.0
                
                # 按步数进行评估
                if global_step % args.eval_steps == 0:
                    print(f'\n--- Evaluation at step {global_step} ---')
                    val_mse, val_mae, val_rmse, val_r2, _, _ = eval_model(
                        model, val_loader, device, criterion, scaler=scaler
                    )
                    
                    print(f'Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | '
                          f'Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}')
                    
                    training_history.append({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'val_mse': float(val_mse),
                        'val_mae': float(val_mae),
                        'val_rmse': float(val_rmse),
                        'val_r2': float(val_r2)
                    })
                    
                    # 保存最佳模型
                    if val_mse < best_val_loss:
                        best_val_loss = val_mse
                        model_path = os.path.join(args.output_dir, args.save_model)
                        torch.save(model.state_dict(), model_path)
                        print(f'✓ Best model saved to {model_path}!')
                    
                    print('-' * 50)
                    model.train()  # 切回训练模式
        
        # Epoch结束时的统计
        avg_epoch_loss = epoch_loss / len(train_loader) * args.gradient_accumulation_steps
        print(f'\nEpoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}\n')
    
    return best_val_loss, training_history

# 5. 评估函数（带平均）
def eval_model(model, dataloader, device, criterion, scaler=None):
    model.eval()
    predictions_by_question = {}
    actuals_by_question = {}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulties = batch['difficulty'].to(device)
            question_ids = batch['question_id'].cpu().numpy()
            
            predictions = model(input_ids, attention_mask)
            
            predictions_np = predictions.cpu().numpy()
            difficulties_np = difficulties.cpu().numpy()
            
            # 按question_id分组
            for qid, pred, actual in zip(question_ids, predictions_np, difficulties_np):
                qid = int(qid)
                if qid not in predictions_by_question:
                    predictions_by_question[qid] = []
                    actuals_by_question[qid] = actual
                predictions_by_question[qid].append(pred)
    
    # 对每个问题的预测取平均
    final_predictions = []
    final_actuals = []
    
    for qid in sorted(predictions_by_question.keys()):
        avg_pred = np.mean(predictions_by_question[qid])
        final_predictions.append(avg_pred)
        final_actuals.append(actuals_by_question[qid])
    
    predictions_array = np.array(final_predictions)
    actuals_array = np.array(final_actuals)
    
    # 反归一化
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
    parser = argparse.ArgumentParser(description='Data Augmentation Transformer Difficulty Regression')
    parser.add_argument('--train_file', type=str, default='Cambridge_train.json')
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default='Cambridge_test.json')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--training_target', type=str, default='difficulty')
    parser.add_argument('--answer_keys', type=str, nargs='+', default=[], 
                        help='A list of keys for the answers to include from the JSON file (e.g., --answer_keys answer1 answer2)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Run evaluation every X steps')
    parser.add_argument('--log_steps', type=int, default=10,
                        help='Log training info every X steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./output_augmented')
    parser.add_argument('--save_model', type=str, default='best_model_augmented.pt')
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
    default_log = os.path.join(args.output_dir, f'train_augmented_{timestamp}.log')
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
    
    print("\nLoading data...")
    train_questions, train_answers, train_difficulties = load_data(args.train_file, args.training_target, args.answer_keys)
    print(f"Loaded training data: {len(train_questions)} questions, with {len(args.answer_keys)} answers per question.")
    
    if args.val_file is not None:
        val_questions, val_answers, val_difficulties = load_data(args.val_file, args.training_target, args.answer_keys)
        print(f"Loaded validation data: {len(val_questions)} questions.")
    else:
        print(f"Splitting validation set from training data...")
        train_questions, val_questions, train_answers, val_answers, train_difficulties, val_difficulties = train_test_split(
            train_questions, train_answers, train_difficulties,
            test_size=args.val_split, 
            random_state=args.seed,
            shuffle=True
        )
        print(f"Split result - Train: {len(train_questions)} questions, Val: {len(val_questions)} questions")
    
    test_questions, test_answers, test_difficulties = load_data(args.test_file, args.training_target, args.answer_keys)
    print(f"Loaded test data: {len(test_questions)} questions.")
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
    
    # 创建数据增强后的数据集
    train_dataset = AugmentedDataset(train_questions, train_answers, train_difficulties, tokenizer, args.max_length)
    val_dataset = AugmentedDataset(val_questions, val_answers, val_difficulties, tokenizer, args.max_length)
    test_dataset = AugmentedDataset(test_questions, test_answers, test_difficulties, tokenizer, args.max_length)
    
    print(f"\nAfter data augmentation:")
    print(f"  Train: {len(train_questions)} questions -> {len(train_dataset)} samples (augmentation factor: {len(train_dataset)/len(train_questions):.2f}x)")
    print(f"  Val: {len(val_questions)} questions -> {len(val_dataset)} samples (augmentation factor: {len(val_dataset)/len(val_questions):.2f}x)")
    print(f"  Test: {len(test_questions)} questions -> {len(test_dataset)} samples (augmentation factor: {len(test_dataset)/len(test_questions):.2f}x)")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleTransformerRegressor(
        model_name=args.model_name, 
        dropout=args.dropout, 
        cache_dir=args.cache_dir
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # 计算total_steps时需要考虑gradient accumulation
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_steps is None else args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print(f"\nScheduler: {total_steps} total update steps, {warmup_steps} warmup steps.")
    print("-" * 50)
    
    # 训练模型
    best_val_loss, training_history = train_model(
        model, train_loader, val_loader, optimizer, device, criterion, 
        scheduler, args, scaler=scaler
    )
    
    print('\n' + '=' * 50)
    print('Evaluating on test set with the best model...')
    print('=' * 50)
    model_path = os.path.join(args.output_dir, args.save_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_mse, test_mae, test_rmse, test_r2, test_predictions, test_actuals = eval_model(
        model, test_loader, device, criterion, scaler=scaler
    )
    
    print(f'\nTest Results (averaged predictions, original scale):')
    print(f'  MSE: {test_mse:.4f}')
    print(f'  MAE: {test_mae:.4f}')
    print(f'  RMSE: {test_rmse:.4f}')
    print(f'  R²: {test_r2:.4f}')
    
    results = {
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'best_val_loss': float(best_val_loss),
        'normalized': args.normalize,
        'num_answer_keys': len(args.answer_keys),
        'augmentation_factor': len(test_dataset) / len(test_questions),
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
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
    
    print('\n' + '=' * 50)
    print('Training completed!')
    print('=' * 50)
    
    if _LOG_FH:
        _LOG_FH.close()

if __name__ == '__main__':
    main()