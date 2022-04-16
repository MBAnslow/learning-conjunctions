import spacy
import subprocess
import sys


def install_spacy_required_packages():
    packages = ['en', 'en_core_web_sm']

    for package_name in packages:
        if not spacy.util.is_package(package_name):
            subprocess.check_call([sys.executable, "-m", "spacy", "download", package_name])

install_spacy_required_packages()


nlp = spacy.load('en_core_web_sm')
single_conj_examples = []

with open("brown.txt") as f:
    paragraph_start = False

    for line in f.readlines():

        doc = nlp(line)

        for sent_doc in doc.sents:
            contains_conj = sum([tok.pos_ in ['CCONJ', "CONJ", "SCONJ"] for tok in sent_doc])
            is_speech = sum([tok.orth_ == '"' for tok in sent_doc]) > 0
            if contains_conj == 1 and len(sent_doc) < 20 and not is_speech and len(sent_doc) > 0:
                single_conj_examples.append(str(sent_doc) + '\n')

with open("single_cconj.txt", 'w') as f:
    f.writelines(single_conj_examples)
