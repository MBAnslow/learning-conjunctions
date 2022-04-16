from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils import sent_scoring

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

lines = []

with open('single_cconj.txt') as f:
    for line in f.readlines():
        lines.append(line.strip('\n'))


import spacy
nlp = spacy.load('en_core_web_sm')

cconj = ["for", "and", "nor", "but", "or", "yet", "so"]
subordinating_conj = ["after", "as", "as long as", "as soon as", "as though", "before", "even if", "if", "if when",
                      "inasmuch", "just as", "now", "now that", "once", "provided that", "since", "supposing",
                      "that", "though", "until", "whenever", "whereas", "wherever", "which", "who"]


tags_predicted = []
scores_original = []
scores_generated = []

for line in lines:
    doc = nlp(line)
    conj = [(idx, tok) for idx, tok in enumerate(doc) if tok.pos_ in ['CCONJ', "CONJ", "SCONJ"]]

    if not conj:
        continue

    idx, tok = conj[0]

    pre = str(doc[:idx])
    pre_out = generator(pre)[0]
    pre_out_doc = nlp(pre_out['generated_text'])
    pos = pre_out_doc[idx].pos_
    sent_doc = None

    for sent_doc in pre_out_doc.sents:
        sent_doc = sent_doc
        break

    score_original = sent_scoring((model, tokenizer), str(line), False)
    score_gen = sent_scoring((model, tokenizer), str(sent_doc), False)

    tags_predicted.append(pos)
    scores_original.append(score_original)
    scores_generated.append(score_gen)
    print(tags_predicted)

a = 1
