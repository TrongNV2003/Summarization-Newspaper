from summarization import Summarization
from summarization import process_files

summa = Summarization()

with open('data/context.txt', 'r',encoding='utf-8') as a:
    article = a.read()

process_files(
    article,
    output_file = "result/output.json"
)