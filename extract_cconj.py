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
examples = 0

with open("data/single_cconj.txt", 'w+') as f_cconj:
    with open("data/gutenberg.txt") as f:
        paragraph_start = False

        for line in f.readlines():
            n = 4000
            chunks = [line[i:i + n] for i in range(0, len(line), n)]

            for doc in nlp.pipe(chunks, n_process=6):

                for sent_doc in doc.sents:
                    contains_conj = [(idx, tok.pos_) for idx, tok in enumerate(sent_doc) if tok.pos_ in ["CCONJ", "SCONJ", "CONJ"]]
                    is_speech = sum([tok.orth_ == '"' for tok in sent_doc]) > 0
                    is_not_fragment = sent_doc[0].orth_[0].isupper() and sent_doc[-1].pos_ == "PUNCT"
                    if len(contains_conj) == 1 and len(sent_doc) < 20 and not is_speech and len(sent_doc) > 0 and is_not_fragment:
                        examples += 1
                        f_cconj.write(str(contains_conj) + ':::' +str(sent_doc) + '\n')
                        print(examples)
