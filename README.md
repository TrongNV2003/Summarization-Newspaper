# Summarization-Finetune-BARTpho

## Clone source code
```bash
!git clone https://github.com/TrongNV2003/Bartpho-summarization-finetune.git
```

## Usage
First, you need to install requirement pakage:

```bash
!pip install -r requirement.txt
```

### Download dataset
You need to get summarization dataset. Get from [HuggingFace](Trongdz/summarization-dataset).

Or clone:
```bash
!git clone https://huggingface.co/datasets/Trongdz/summarization-dataset
``` 
(Dataset should put inside source code)

### Fine-tune model
```bash
!python training/train_qa.py
```

### Test model
```bash
!python training/test_qa.py
```

### Evaluate model
```bash
!python rouge_score.py
```