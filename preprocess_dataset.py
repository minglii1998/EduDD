import json
import sys

dict_lowercase = {
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
}

def preprocess_dataset(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    data_new = []

    for item in data:
        for question_key in item['questions']:
            question = item['questions'][question_key]

            if question['text'] == '':
                continue
            
            text = f"Reading Comprehension Question: {question['text']}\nOptions:\n(A) {question['options']['a']['text']}\n(B) {question['options']['b']['text']}\n(C) {question['options']['c']['text']}\n(D) {question['options']['d']['text']}\nCorrect Answer: ({dict_lowercase[question['answer']]}) {question['options'][question['answer']]['text']}\nReference Passage: {item['text']}\nCommon European Framework of Reference for Languages (CEFR) Level: {item['level']}"

            data_new.append({
                'processed_text': text,
                'difficulty': question['diff'],
                'discrimination': question['disc'],
                'facility': question['fac'],
                'answer': question['answer'],
                'original_question_dict': question,
                'original_passage_id': item['id'],
            })

    with open(output_file, 'w') as f:
        json.dump(data_new, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    preprocess_dataset('split_pub/Cambridge_mcq_test_pub.json', 'split_pub/Cambridge_mcq_test_pub_Processed.json')
    preprocess_dataset('split_pub/Cambridge_mcq_train_pub.json', 'split_pub/Cambridge_mcq_train_pub_Processed.json')