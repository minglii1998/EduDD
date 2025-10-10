import json
import sys
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
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

_LOG_FH = None  # 保持文件句柄存活，防止被回收

# ========== 新增：Scalar Mix Layer ==========
class ScalarMix(nn.Module):
    """
    Scalar Mix Layer: 对所有层的hidden states进行加权融合
    参考论文: "Deep Contextualized Word Representations" (Peters et al., 2018)
    
    注意：即使某些层被冻结，scalar weights 仍然是可学习的
    因为我们是在学习"如何组合"已有的表示，而不是改变表示本身
    """
    def __init__(self, num_layers, trainable=True, use_gamma=True):
        super(ScalarMix, self).__init__()
        self.num_layers = num_layers
        self.trainable = trainable
        self.use_gamma = use_gamma
        
        # 为每一层创建一个可学习的标量权重
        if trainable:
            # 初始化为均匀权重（经过log后）
            self.scalar_weights = nn.Parameter(torch.zeros(num_layers))
        else:
            # 如果不可训练，使用均匀权重
            self.register_buffer('scalar_weights', torch.zeros(num_layers))
        
        # Gamma参数用于缩放混合后的表示（可选）
        if use_gamma:
            if trainable:
                self.gamma = nn.Parameter(torch.tensor(1.0))
            else:
                self.register_buffer('gamma', torch.tensor(1.0))
        else:
            self.gamma = 1.0
    
    def forward(self, hidden_states_list):
        """
        Args:
            hidden_states_list: List of tensors, each of shape [batch_size, seq_len, hidden_size]
                               长度为 num_layers
        
        Returns:
            mixed_hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            weights: Tensor of shape [num_layers] - softmax normalized weights
        """
        if len(hidden_states_list) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden states, got {len(hidden_states_list)}")

        weights = torch.softmax(self.scalar_weights, dim=0)     # [L]
        mixed = None
        for i, hs in enumerate(hidden_states_list):             # 每个 hs: [B,T,H]
            term = hs * weights[i]
            mixed = term if mixed is None else (mixed + term)   # 累加，避免stack
        mixed = (self.gamma * mixed) if not isinstance(self.gamma, float) else (self.gamma * mixed)

        return mixed, weights

# 1. 自定义数据集类（适配Decoder-Only模型）
class DifficultyDataset(Dataset):
    def __init__(self, texts, difficulties, tokenizer, max_length=512, system_prompt=None, prefix_prompt=''):
        self.texts = texts
        self.difficulties = difficulties
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.prefix_prompt = prefix_prompt

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        difficulty = self.difficulties[idx]
        
        # 使用chat template格式化输入
        if self.system_prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prefix_prompt + text}
                ]
        else:
            messages = [
                {"role": "user", "content": self.prefix_prompt + text}
                ]
        
        # 应用chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 编码文本
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
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

# 3. 定义Decoder-Only回归模型（支持LLaMA/Mistral等）+ Scalar Mix
class DecoderOnlyRegressor(nn.Module):
    def __init__(self, model_name='meta-llama/Llama-2-7b-chat-hf', dropout=0.3, use_fp16=False, 
                 freeze_layers=None, pooling_method='last', cache_dir='/nfshomes/minglii/scratch/cache/hub',
                 use_scalar_mix=False, scalar_mix_layer_indices=None,
                 scalar_mix_trainable=True, scalar_mix_use_gamma=True):
        super(DecoderOnlyRegressor, self).__init__()
        self.model_name = model_name
        self.pooling_method = pooling_method  # 'last' or 'mean'
        self.use_scalar_mix = use_scalar_mix
        
        if pooling_method not in ['last', 'mean']:
            raise ValueError(f"pooling_method must be 'last' or 'mean', got '{pooling_method}'")
        
        cache_directory = os.path.expanduser(cache_dir)
        
        # 根据参数选择精度
        dtype = torch.float16 if use_fp16 else torch.float32
        
        # 加载decoder-only模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_directory,
                torch_dtype=dtype,
                device_map=None  # 不自动分配设备，手动控制
            )
        except OSError:
            print(f"PyTorch weights not found, loading from TensorFlow weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                from_tf=True, 
                cache_dir=cache_directory,
                torch_dtype=dtype,
                device_map=None
            )
        
        # 冻结LM head（我们不会使用它，只需要hidden states）
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = False
        
        # 获取模型层数（不包括embedding层）
        self.num_transformer_layers = len(self.model.model.layers)
        
        # 冻结大部分层（如果指定）
        self.frozen_layer_indices = set()
        if freeze_layers is not None:
            self._freeze_layers(num_layers_to_train=freeze_layers)
        
        # ========== Scalar Mix配置 ==========
        if use_scalar_mix:
            # 确定要使用哪些层
            if scalar_mix_layer_indices is not None:
                # 用户指定了要使用的层索引
                self.scalar_mix_layer_indices = scalar_mix_layer_indices
            else:
                # 默认使用所有transformer层（不包括embedding层）
                # 索引从0开始，0表示第一个transformer层的输出
                self.scalar_mix_layer_indices = list(range(self.num_transformer_layers))
            
            num_layers_for_mix = len(self.scalar_mix_layer_indices)
            
            # 创建Scalar Mix层
            # 重要：即使某些层被冻结，scalar mix的权重仍然可以训练
            # 因为我们是在学习"如何组合"已有的表示，而不是改变表示本身
            self.scalar_mix = ScalarMix(
                num_layers=num_layers_for_mix,
                trainable=scalar_mix_trainable,
                use_gamma=scalar_mix_use_gamma
            )
            
            print(f"\n{'='*50}")
            print(f"Scalar Mix Configuration:")
            print(f"  Using {num_layers_for_mix} layers: {self.scalar_mix_layer_indices}")
            print(f"  Frozen transformer layers: {sorted(self.frozen_layer_indices) if self.frozen_layer_indices else 'None'}")
            print(f"  Scalar Mix trainable: {scalar_mix_trainable}")
            print(f"  Use gamma scaling: {scalar_mix_use_gamma}")
            print(f"{'='*50}\n")
        
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)
    
    def _freeze_layers(self, num_layers_to_train=4):
        """冻结除了最后几层外的所有参数"""
        # 冻结embedding层
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False
        
        # 冻结前面的transformer层
        total_layers = self.num_transformer_layers
        for i, layer in enumerate(self.model.model.layers):
            if i < total_layers - num_layers_to_train:
                for param in layer.parameters():
                    param.requires_grad = False
                self.frozen_layer_indices.add(i)
        
        print(f"Frozen all but last {num_layers_to_train} layers")
        print(f"Frozen layer indices: {sorted(self.frozen_layer_indices)}")
    
    def forward(self, input_ids, attention_mask):
        # 获取模型输出，包括所有层的hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # outputs.hidden_states是一个tuple，包含:
        # - hidden_states[0]: embedding层的输出
        # - hidden_states[1:]: 每个transformer层的输出
        # 总共有 num_transformer_layers + 1 个元素
        
        if self.use_scalar_mix:
            # 提取指定层的hidden states
            selected_hidden_states = [
                outputs.hidden_states[idx + 1]  # +1因为索引0是embedding层
                for idx in self.scalar_mix_layer_indices
            ]
            
            # 使用Scalar Mix融合
            mixed_hidden_states, layer_weights = self.scalar_mix(selected_hidden_states)
            # mixed_hidden_states: [batch_size, seq_len, hidden_size]
            
            # 保存权重用于分析（可选）- 立即detach并移到CPU
            self.last_layer_weights = layer_weights.detach().cpu()
            
            # 使用融合后的hidden states
            hidden_states = mixed_hidden_states
            
            # 清理不需要的hidden states以释放内存
            del selected_hidden_states, outputs.hidden_states
        else:
            # 原始方式：只使用最后一层的hidden states
            hidden_states = outputs.hidden_states[-1]
            # 清理不需要的hidden states
            del outputs.hidden_states
        
        # Pooling操作
        if self.pooling_method == 'last':
            # 找到每个样本中最后一个非padding token的位置
            sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            batch_size = hidden_states.shape[0]
            pooled_output = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        # elif self.pooling_method == 'mean':
        #     # Mean pooling
        #     attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        #     sum_hidden_states = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        #     sum_mask = attention_mask_expanded.sum(dim=1)
        #     sum_mask = torch.clamp(sum_mask, min=1e-9)
        #     pooled_output = sum_hidden_states / sum_mask

        elif self.pooling_method == 'mean':
            # Mean pooling
            mask = attention_mask.unsqueeze(-1).float()       # [B, T, 1]
            summed = (hidden_states * mask).sum(dim=1)         # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-9)           # [B, 1]
            pooled_output = summed / counts                    # [B, H]

        
        # 应用dropout和回归层
        output = self.dropout(pooled_output)
        output = self.regressor(output)
        
        return output.squeeze(-1)
    
    def get_layer_weights(self):
        """获取Scalar Mix的层权重（用于分析）"""
        if self.use_scalar_mix and hasattr(self, 'last_layer_weights'):
            return self.last_layer_weights.numpy()
        return None

# 4. 训练函数
def train_epoch(model, dataloader, optimizer, device, criterion, scheduler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    print(f"  Training on {len(dataloader)} batches...")
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        difficulties = batch['difficulty'].to(device)
        
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, difficulties)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
            
            # 定期清理GPU缓存
            if (step + 1) % (accumulation_steps * 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_loss += loss.item() * accumulation_steps
        
        # # 每10个batch打印一次进度
        # if (step + 1) % 10 == 0:
        #     print(f"    Batch {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    print(f"  Training completed. Average loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)

# 5. 评估函数
def eval_model(model, dataloader, device, criterion, scaler=None, return_layer_weights=False):
    """评估模型"""
    model.eval()
    predictions_list = []
    actuals_list = []
    layer_weights_list = []
    
    print(f"  Evaluating on {len(dataloader)} batches...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulties = batch['difficulty'].to(device)
            
            predictions = model(input_ids, attention_mask)
            
            predictions_np = predictions.cpu().numpy()
            predictions_list.extend(np.atleast_1d(predictions_np))
            actuals_list.extend(np.atleast_1d(difficulties.cpu().numpy()))
            
            # 收集layer weights
            if return_layer_weights and model.use_scalar_mix:
                weights = model.get_layer_weights()
                if weights is not None:
                    layer_weights_list.append(weights)
    
    predictions_array = np.array(predictions_list)
    actuals_array = np.array(actuals_list)
    
    # 检查是否有NaN或Inf值
    if np.any(np.isnan(predictions_array)) or np.any(np.isinf(predictions_array)):
        print("Warning: Found NaN or Inf in predictions!")
        print(f"  NaN count: {np.sum(np.isnan(predictions_array))}")
        print(f"  Inf count: {np.sum(np.isinf(predictions_array))}")
        valid_mask = ~(np.isnan(predictions_array) | np.isinf(predictions_array))
        if np.any(valid_mask):
            mean_val = np.mean(predictions_array[valid_mask])
            predictions_array[~valid_mask] = mean_val
        else:
            predictions_array = np.zeros_like(predictions_array)
    
    # 如果提供了scaler，转换回原始scale
    if scaler is not None:
        predictions_array = scaler.inverse_transform(predictions_array.reshape(-1, 1)).flatten()
        actuals_array = scaler.inverse_transform(actuals_array.reshape(-1, 1)).flatten()
    
    # 计算所有指标
    mse = np.mean((predictions_array - actuals_array) ** 2)
    mae = np.mean(np.abs(predictions_array - actuals_array))
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_array, predictions_array)
    
    result = (mse, mae, rmse, r2, predictions_array.tolist(), actuals_array.tolist())
    
    if return_layer_weights and len(layer_weights_list) > 0:
        # 平均所有batch的layer weights
        avg_layer_weights = np.mean(layer_weights_list, axis=0)
        return result + (avg_layer_weights,)
    
    return result

# 6. 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Decoder-Only Model Difficulty Regression Training with Scalar Mix')
    
    # 数据相关参数
    parser.add_argument('--train_file', type=str, default='Cambridge_train.json', 
                        help='Path to training data file')
    parser.add_argument('--val_file', type=str, default=None, 
                        help='Path to validation data file (optional)')
    parser.add_argument('--test_file', type=str, default='Cambridge_test.json', 
                        help='Path to test data file')
    parser.add_argument('--val_split', type=float, default=0.15, 
                        help='Validation split ratio (0-1)')
    parser.add_argument('--training_target', type=str, default='difficulty', 
                        help='difficulty, discrimination, facility')
    parser.add_argument('--system_prompt', type=str, 
                        default=None,
                        help='System prompt for chat template')
    parser.add_argument('--prefix_prompt', type=str, 
                        default='',
                        help='Prefix prompt for chat template')

    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
                        help='Model name (e.g., meta-llama/Llama-2-7b-chat-hf, mistralai/Mistral-7B-Instruct-v0.2)')
    parser.add_argument('--max_length', type=int, default=512, 
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.3, 
                        help='Dropout rate')
    parser.add_argument('--pooling_method', type=str, default='last', choices=['last', 'mean'],
                        help='Pooling method for hidden states: "last" (last token) or "mean" (mean pooling)')
    
    # ========== 新增：Scalar Mix参数 ==========
    parser.add_argument('--use_scalar_mix', action='store_true',
                        help='Use Scalar Mix to combine hidden states from multiple layers')
    parser.add_argument('--scalar_mix_layers', type=str, default=None,
                        help='Comma-separated layer indices to use for Scalar Mix (e.g., "0,5,10,15"). '
                             'If not specified, all layers will be used.')
    parser.add_argument('--scalar_mix_trainable', action='store_true', default=True,
                        help='Whether scalar mix weights are trainable (default: True)')
    parser.add_argument('--scalar_mix_use_gamma', action='store_true', default=True,
                        help='Whether to use gamma scaling parameter in scalar mix (default: True)')
    parser.add_argument('--no_scalar_mix_trainable', dest='scalar_mix_trainable', action='store_false',
                        help='Disable training of scalar mix weights')
    parser.add_argument('--no_scalar_mix_gamma', dest='scalar_mix_use_gamma', action='store_false',
                        help='Disable gamma scaling in scalar mix')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Batch size (smaller for decoder-only models)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, 
                        help='Ratio of total steps for learning rate warmup')
    parser.add_argument('--warmup_steps', type=int, default=None, 
                        help='Number of warmup steps (overrides warmup_ratio if set)')
    parser.add_argument('--use_fp16', action='store_true',
                        help='Use mixed precision training (may cause NaN issues)')
    parser.add_argument('--freeze_layers', type=int, default=None,
                        help='Number of last layers to train (freeze the rest)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output_decoder', 
                        help='Directory to save model and results')
    parser.add_argument('--save_model', type=str, default='best_model_decoder.pt', 
                        help='Filename for saving the best model')
    parser.add_argument('--normalize', action='store_true', 
                        help='Apply z-score normalization to difficulty values')
    parser.add_argument('--cache_dir', type=str, default='/nfshomes/minglii/scratch/cache/hub',
                        help='Cache directory for model downloads (default: /nfshomes/minglii/scratch/cache/hub)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (default: <output_dir>/train_phi_YYYYMMDD_HHMMSS.log)')
    
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
    
    # 解析scalar_mix_layers参数
    scalar_mix_layer_indices = None
    if args.scalar_mix_layers is not None:
        scalar_mix_layer_indices = [int(x.strip()) for x in args.scalar_mix_layers.split(',')]
        print(f"Using specified layers for Scalar Mix: {scalar_mix_layer_indices}")
    
    # ========== 日志重定向 ==========
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    default_log = os.path.join(args.output_dir, f'train_phi_{timestamp}.log')
    log_path = args.log_file if args.log_file is not None else default_log
    
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Removed existing log file: {log_path}")
    
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
    
    # ========== 加载数据 ==========
    print("\nLoading data...")
    
    train_texts, train_difficulties = load_data(args.train_file, args.training_target)
    print(f"Loaded training data: {len(train_texts)} samples")
    
    if args.val_file is not None:
        val_texts, val_difficulties = load_data(args.val_file, args.training_target)
        print(f"Loaded validation data: {len(val_texts)} samples")
    else:
        print(f"No validation file provided, splitting {args.val_split*100:.0f}% from training data...")
        train_texts, val_texts, train_difficulties, val_difficulties = train_test_split(
            train_texts, train_difficulties, 
            test_size=args.val_split, 
            random_state=args.seed,
            shuffle=True
        )
        print(f"Split result - Train: {len(train_texts)} samples, Val: {len(val_texts)} samples")
    
    test_texts, test_difficulties = load_data(args.test_file, args.training_target)
    print(f"Loaded test data: {len(test_texts)} samples")
    print("-" * 50)
    
    # ========== Z-score标准化 ==========
    scaler = None
    if args.normalize:
        print("\nApplying z-score normalization to difficulty values...")
        scaler = StandardScaler()
        
        train_difficulties_array = np.array(train_difficulties).reshape(-1, 1)
        scaler.fit(train_difficulties_array)
        
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    train_dataset = DifficultyDataset(train_texts, train_difficulties, tokenizer, 
                                     args.max_length, args.system_prompt, args.prefix_prompt)
    val_dataset = DifficultyDataset(val_texts, val_difficulties, tokenizer, 
                                   args.max_length, args.system_prompt, args.prefix_prompt)
    test_dataset = DifficultyDataset(test_texts, test_difficulties, tokenizer, 
                                    args.max_length, args.system_prompt, args.prefix_prompt)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading model...")
    model = DecoderOnlyRegressor(
        model_name=args.model_name, 
        dropout=args.dropout,
        use_fp16=args.use_fp16,
        freeze_layers=args.freeze_layers,
        pooling_method=args.pooling_method,
        cache_dir=args.cache_dir,
        use_scalar_mix=args.use_scalar_mix,
        scalar_mix_layer_indices=scalar_mix_layer_indices,
        scalar_mix_trainable=args.scalar_mix_trainable,
        scalar_mix_use_gamma=args.scalar_mix_use_gamma
    ).to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Pooling method: {args.pooling_method}")
    print(f"Scalar Mix enabled: {args.use_scalar_mix}")
    if args.use_fp16:
        print("⚠️  Using FP16 - watch for NaN values during training")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # ========== 创建learning rate scheduler ==========
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nLearning rate scheduler:")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
    print(f"  Initial LR: {args.lr}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print("-" * 50)
    
    best_val_loss = float('inf')
    training_history = []
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n=== Starting Epoch {epoch+1}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, 
                                scheduler=scheduler, accumulation_steps=args.accumulation_steps)
        
        # 评估时收集layer weights
        eval_result = eval_model(model, val_loader, device, criterion, scaler=scaler, 
                                return_layer_weights=args.use_scalar_mix)
        
        if args.use_scalar_mix and len(eval_result) == 7:
            val_mse, val_mae, val_rmse, val_r2, _, _, layer_weights = eval_result
            # 打印layer weights
            print(f'\nEpoch {epoch+1}/{args.epochs} - Layer Weights:')
            for i, (layer_idx, weight) in enumerate(zip(model.scalar_mix_layer_indices, layer_weights)):
                print(f'  Layer {layer_idx}: {weight:.4f}')
        else:
            val_mse, val_mae, val_rmse, val_r2, _, _ = eval_result
        
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'\nEpoch {epoch+1}/{args.epochs}')
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
            
            # 清理GPU缓存并强制垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # 保存模型
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ Best model saved to {model_path}!')
            
            # 再次清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        print('-' * 50)
    
    print('\nEvaluating on test set...')
    model_path = os.path.join(args.output_dir, args.save_model)
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 测试时也收集layer weights
    test_result = eval_model(model, test_loader, device, criterion, scaler=scaler, 
                            return_layer_weights=args.use_scalar_mix)
    
    if args.use_scalar_mix and len(test_result) == 7:
        test_mse, test_mae, test_rmse, test_r2, test_predictions, test_actuals, test_layer_weights = test_result
        
        print(f'\nFinal Layer Weights on Test Set:')
        for i, (layer_idx, weight) in enumerate(zip(model.scalar_mix_layer_indices, test_layer_weights)):
            print(f'  Layer {layer_idx}: {weight:.4f}')
        print()
    else:
        test_mse, test_mae, test_rmse, test_r2, test_predictions, test_actuals = test_result
        test_layer_weights = None
    
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
        'pooling_method': args.pooling_method,
        'use_scalar_mix': args.use_scalar_mix,
        'training_history': training_history,
        'predictions': [float(p) for p in test_predictions],
        'actuals': [float(a) for a in test_actuals]
    }
    
    if args.normalize:
        results['scaler_mean'] = float(scaler.mean_[0])
        results['scaler_std'] = float(scaler.scale_[0])
    
    if args.use_scalar_mix and test_layer_weights is not None:
        results['scalar_mix_layer_indices'] = model.scalar_mix_layer_indices
        results['final_layer_weights'] = {
            f'layer_{idx}': float(weight) 
            for idx, weight in zip(model.scalar_mix_layer_indices, test_layer_weights)
        }
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nTest results saved to {results_path}')
    
    print('\nSample predictions on test set:')
    for i in range(min(5, len(test_predictions))):
        print(f'  Sample {i+1}: Predicted={test_predictions[i]:.4f}, Actual={test_actuals[i]:.4f}')
    
    print('\nTraining completed!')
    
    # 关闭日志文件
    if _LOG_FH is not None:
        _LOG_FH.close()

if __name__ == '__main__':
    main()