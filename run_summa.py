from summarization import Summarization

summa = Summarization()

with open('data/context.txt', 'r',encoding='utf-8') as a:
    article = a.read()

summa.process_files(
    article,
    output_file = "result/output.json"
)