# Finetune model for Summarization

In this project, i use model BARTpho and ViT5.
## Clone source code
```bash
git clone https://github.com/TrongNV2003/Bartpho-summarization-finetune.git
```

## Usage
First, you need to install requirement pakage:
```bash
pip install -r requirement.txt
```

### Clone Newspaper dataset
You need to clone summarization dataset. Get from [HuggingFace](https://huggingface.co/datasets/Trongdz/summarization-dataset).

Or clone from huggingface:
```bash
git clone https://huggingface.co/datasets/Trongdz/summarization-dataset
``` 
(Dataset should put inside source code folder path)

### Fine-tune model
For fine-tune model, run:
```bash
python training/train_qa.py
```
Or can get pre-trained model: 
-BARTpho: [HuggingFace](https://huggingface.co/Trongdz/bartpho-summarization)
-ViT5: [HuggingFace](https://huggingface.co/Trongdz/ViT5-summarization)

### Test model
```bash
python training/test_qa.py
```

### Evaluate model
```bash
python rouge_score.py
```

### Run model
You can change any context in "data/context.txt" for summarizing, then run:
```bash
python run_summa.py
```