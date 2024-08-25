from rouge_score import rouge_scorer
import json

def calculate_rouge_score(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Khởi tạo điểm tổng hợp
    aggregate_scores = {
        "rouge1": {"fmeasure": 0},
        "rouge2": {"fmeasure": 0},
        "rougeL": {"fmeasure": 0},
    }

    # Duyệt qua từng cặp dự đoán và tham chiếu
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)

        for key in aggregate_scores.keys():

            aggregate_scores[key]["fmeasure"] += scores[key].fmeasure

    # Tính điểm trung bình cho mỗi metric
    num_samples = len(predictions)
    for key in aggregate_scores.keys():
        aggregate_scores[key]["fmeasure"] /= num_samples

    return aggregate_scores


with open('summarization_results.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

contexts = [item['context'] for item in data]
summaries = [item['summary'] for item in data]

with open('reference_text.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

reference_summary = [item['reference_summary'] for item in data]

# Tính điểm ROUGE
rouge_scores = calculate_rouge_score(summaries, reference_summary)

for metric, scores in rouge_scores.items():
    print(f"{metric.upper()}: {scores['fmeasure']:.10f}")