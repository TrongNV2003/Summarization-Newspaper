import os
import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

class Summarization:
    def __init__(self) -> None:
        self.SEQ_LENGTH = 512
        QG_PRETRAINED = "vinai/bartpho-word-base"
        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED)
        self.qg_model.to(self.device)
        self.qg_model.eval()

    def remove_last_line(self, text: str) -> str:
        # Tách văn bản thành các dòng
        lines = text.strip().split("\n")
        # Loại bỏ dòng cuối cùng
        return "\n".join(lines[:-1]).strip()

    def extract_summary_line(self, text: str) -> Tuple[str, str]:
        # Tách văn bản thành các dòng
        lines = text.strip().split("\n")
        # Lấy dòng cuối cùng làm câu tóm tắt
        summary_line = lines[-1].strip()
        # # Ghép lại các dòng trước đó
        # context = "\n".join(lines[:-1]).strip()
        return summary_line

    def summarize(self, text: str) -> str:
        # Chuẩn bị văn bản (không bao gồm dòng cuối cùng)
        text_without_last_line = self.remove_last_line(text)

        inputs = self.qg_tokenizer(text_without_last_line, max_length=self.SEQ_LENGTH, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        summary_ids = self.qg_model.generate(inputs["input_ids"], max_length=30, num_beams=4)
        summary = self.qg_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def process_files(input_folder: str, output_file: str, reference_file: str) -> None:
    summarizer = Summarization()
    results = []
    reference_text = []

    # Lấy danh sách tất cả các tệp tin .txt.seg trong thư mục
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt.seg")]


    for file_name in tqdm(files, desc="Processing files"):
        file_path = os.path.join(input_folder, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()

            if text:
                summary = summarizer.summarize(text)
                summary_line = summarizer.extract_summary_line(text)
                results.append({
                    "file_name": file_name,
                    "context": text,
                    "summary": summary
                })

                reference_text.append({
                    "file_name": file_name,
                    "reference_summary": summary_line
                })
            else:
                print(f"Warning: {file_name} is empty and will be skipped.")

    # Lưu kết quả vào tệp JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    with open(reference_file, 'w', encoding='utf-8') as ref_file:
        json.dump(reference_text, ref_file, ensure_ascii=False, indent=4)

    print(f"Summarization completed. Results saved to {output_file}")
    print(f"Reference test saved to {reference_file}")

if __name__ == "__main__":
    input_folder = "/content/vietnews/data/test_tokenized"
    output_file = "summarization_results.json"
    reference_file = "reference_text.json"
    process_files(input_folder, output_file, reference_file)