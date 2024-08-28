import os
import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

class Summarization:
    def __init__(self) -> None:
        self.SEQ_LENGTH = 512
        QG_PRETRAINED = "Trongdz/bartpho-summarization" #or any other model you want to use
        self.qg_tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED, torch_dtype=torch.bfloat16)
        self.qg_model.to(self.device)
        self.qg_model.eval()

    
    def _split_context(self, text: str) -> List[str]:
        """Splits a long text into segments short enough to be input into the transformer network.
        Segments are used as context for question generation.
        """
        MAX_TOKENS = 512
        paragraphs = text.split("\n")
        tokenized_paragraphs = [self.qg_tokenizer(p)["input_ids"] for p in paragraphs if len(p) > 0]
        segments = []

        while tokenized_paragraphs:
            segment = []

            while len(segment) < MAX_TOKENS and tokenized_paragraphs:
                paragraph = tokenized_paragraphs.pop(0)
                segment.extend(paragraph)
            segments.append(segment)

        return [self.qg_tokenizer.decode(s, skip_special_tokens=True) for s in segments]

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
        split_text = self._split_context(text)

        inputs = self._encode_qg_input(split_text)
        summary_ids = self.qg_model.generate(inputs["input_ids"], max_new_tokens=30, num_beams=4)
        summary = self.qg_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def process_files(text: str, output_file: str) -> None:
    summarizer = Summarization()
    results = []
    summary = summarizer.summarize(text)
    results.append({
        "context": text,
        "summary": summary
    })


    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"Summarization completed. Results saved to {output_file}")