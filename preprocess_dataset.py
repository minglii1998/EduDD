import json
import sys

def preprocess_dataset(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    data_new = []

    for item in data:
        for question_key in item['questions']:
            question = item['questions'][question_key]
            text = f"{question['text']}\nOptions:\n(A) {question['options']['a']['text']}\n(B) {question['options']['b']['text']}\n(C) {question['options']['c']['text']}\n(D) {question['options']['d']['text']}\nReference Passage: {item['text']}"

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
    preprocess_dataset('Cambridge_mcq_test.json', 'Cambridge_mcq_test_Processed.json')
    preprocess_dataset('Cambridge_mcq_train.json', 'Cambridge_mcq_train_Processed.json')