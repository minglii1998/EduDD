import json
import sys
import os
import glob

dict_lowercase = {
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
}

# 全局模型列表 - 只保留这些模型的结果
ALLOWED_MODELS = [
    'Gemma2_2B',
    'Phi35MiniInst',
    'Phi4MiniInst',
    'Qwen25_3B_Inst',
    'Qwen25_7B_Inst',
]

def load_model_results(model_results_dir, allowed_models=None):
    """加载所有模型的结果文件（包括训练和测试文件）
    
    Args:
        model_results_dir: 模型结果文件目录
        allowed_models: 允许的模型名称列表，如果为None则加载所有模型
    """
    model_results = {}
    
    # 获取所有模型结果文件（包括训练和测试）
    test_pattern = os.path.join(model_results_dir, "Cambridge_*_test_results.jsonl")
    train_pattern = os.path.join(model_results_dir, "Cambridge_*_train_results.jsonl")
    
    test_files = glob.glob(test_pattern)
    train_files = glob.glob(train_pattern)
    
    all_files = test_files + train_files
    
    for file_path in all_files:
        # 从文件名提取模型名称
        filename = os.path.basename(file_path)
        if "_test_results.jsonl" in filename:
            model_name = filename.replace("Cambridge_", "").replace("_test_results.jsonl", "")
        elif "_train_results.jsonl" in filename:
            model_name = filename.replace("Cambridge_", "").replace("_train_results.jsonl", "")
        else:
            continue
        
        # 如果指定了允许的模型列表，检查当前模型是否在列表中
        if allowed_models is not None and model_name not in allowed_models:
            print(f"Skipping model {model_name} (not in allowed list)")
            continue
        
        # 读取模型结果
        results = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 使用processed_text作为key来匹配
                key = data['processed_text']
                results[key] = data['model_response']
        
        # 如果模型已经存在，合并结果
        if model_name in model_results:
            model_results[model_name].update(results)
            print(f"Updated {model_name} with {len(results)} additional results from {os.path.basename(file_path)}")
        else:
            model_results[model_name] = results
            print(f"Loaded {len(results)} results for model: {model_name} from {os.path.basename(file_path)}")
    
    return model_results

def preprocess_dataset(input_file, output_file, model_results_dir, allowed_models=None):
    # 加载模型结果
    model_results = load_model_results(model_results_dir, allowed_models)
    
    with open(input_file, 'r') as f:
        data = json.load(f)

    data_new = []

    for item in data:
        for question_key in item['questions']:
            question = item['questions'][question_key]

            if question['text'] == '':
                continue
            
            # 生成用于匹配的文本（不包含Correct Answer部分）
            text_for_matching = f"{question['text']}\nOptions:\n(A) {question['options']['a']['text']}\n(B) {question['options']['b']['text']}\n(C) {question['options']['c']['text']}\n(D) {question['options']['d']['text']}\nReference Passage: {item['text']}\nCommon European Framework of Reference for Languages (CEFR) Level: {item['level']}"
            
            # 生成完整的文本（包含Correct Answer部分）
            text = f"{text_for_matching}\n\nCorrect Answer: ({dict_lowercase[question['answer']]}) {question['options'][question['answer']]['text']}"

            # 创建新的数据项
            new_item = {
                'processed_text': text,
                'difficulty': question['diff'],
                'discrimination': question['disc'],
                'facility': question['fac'],
                'answer': question['answer'],
                'original_question_dict': question,
                'original_passage_id': item['id'],
            }
            
            # 添加所有模型的答案
            for model_name, results in model_results.items():
                model_answer_key = f"{model_name}_answer"
                if text_for_matching in results:
                    new_item[model_answer_key] = results[text_for_matching]
                else:
                    new_item[model_answer_key] = None
                    print(f"Warning: No result found for model {model_name} and question: {question['text'][:50]}...")

            data_new.append(new_item)

    with open(output_file, 'w') as f:
        json.dump(data_new, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    model_results_dir = 'EduAgent_related/model_results'
    
    # 使用全局定义的模型列表
    preprocess_dataset('split_pub/Cambridge_mcq_test_pub.json', 'EduAgent_related/Cambridge_mcq_test_pub_QandA.json', model_results_dir, ALLOWED_MODELS)
    preprocess_dataset('split_pub/Cambridge_mcq_train_pub.json', 'EduAgent_related/Cambridge_mcq_train_pub_QandA.json', model_results_dir, ALLOWED_MODELS)