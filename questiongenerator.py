import os
import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

class Summarization:
    def __init__(self) -> None:
        self.SEQ_LENGTH = 512
        QG_PRETRAINED = "Trongdz/bartpho-summarization"
        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED, torch_dtype=torch.bfloat16)
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

    def _encode_qg_input(self, qg_input: str) -> torch.Tensor:
        """Tokenizes a string and returns a tensor of input ids."""
        return self.qg_tokenizer(
            qg_input,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def summarize(self, text: str) -> str:
        text_without_last_line = self.remove_last_line(text)

        inputs = self._encode_qg_input(text_without_last_line)
        summary_ids = self.qg_model.generate(inputs["input_ids"], max_new_tokens=30, num_beams=4)
        summary = self.qg_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def process_files(input_folder: str, output_file: str, reference_file: str) -> None:
    summarizer = Summarization()
    results = []
    reference_text = []

    # # Lấy danh sách tất cả các tệp tin .txt.seg trong thư mục
    # files = [f for f in os.listdir(input_folder) if f.endswith(".txt.seg")]

    # for file_name in tqdm(files, desc="Processing files"):
    #     file_path = os.path.join(input_folder, file_name)

    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         text = file.read().strip()

    with open(input_folder, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in tqdm(data, desc="Processing files"):
        if 'context' in item:
            text = item['context'].strip()
            if text:
                summary = summarizer.summarize(text)
                summary_line = summarizer.extract_summary_line(text)
                results.append({
                    "context": text,
                    "summary": summary
                })

                reference_text.append({
                    "context": text,
                    "reference_summary": summary_line
                })
            else:
                print(f"Warning empty and will be skipped.")

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    with open(reference_file, 'w', encoding='utf-8') as ref_file:
        json.dump(reference_text, ref_file, ensure_ascii=False, indent=4)

    print(f"Summarization completed. Results saved to {output_file}")
    print(f"Reference test saved to {reference_file}")

if __name__ == "__main__":
    input_folder = "summarization-dataset/test_tokenized.json"
    output_file = "summarization_results.json"
    reference_file = "reference_text.json"
    process_files(input_folder, output_file, reference_file)