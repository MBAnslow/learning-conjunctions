import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import Counter
import spacy

from utils import get_reading_level

nlp = spacy.load("en_core_web_sm")

class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        print('parsing sentences')
        self.sentences = self.load_sentences()
        print('getting unique words')
        self.uniq_words = self.get_uniq_words()
        self.uniq_words = ['PADDING', 'UKNOWN'] + self.uniq_words
        print("Num words: ", len(self.uniq_words))
        self.window = args.window
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

    def load_sentences(self):
        with open("data/single_cconj.txt") as f:

            lines = f.readlines()
            lines = [line.split(":::")[1].lower().strip('\n') for line in lines]
            lines = [line for line in lines if get_reading_level(line) in {"very_easy", "easy", "fairly_easy"}]

        return [" ".join([tok.orth_ for tok in nlp(line)]) for line in lines]

    def get_uniq_words(self):
        word_counts = Counter(" ".join(self.sentences).split(" "))
        return sorted({word: count for word, count in word_counts.items() if word_counts[word] > 4}, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):

        sentence = self.sentences[index].split(" ")
        sentence = [word if word in self.uniq_words else 'UKNOWN' for word in sentence]
        sent_length = min(len(sentence)-1, self.window)
        input = torch.zeros(self.window, dtype=torch.long)
        output = torch.zeros(self.window, dtype=torch.long)
        input[:sent_length] = torch.tensor([self.word_to_index[word] for word in sentence[:sent_length]])
        output[:sent_length] = torch.tensor([self.word_to_index[word] for word in sentence[1:sent_length+1]])
        return (input, output)


class TextGenerator(nn.Module):
    def __init__(self, args, vocab_size):
        super(TextGenerator, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = args.window

        # Embedding layer
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0).to(device)

        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True, dropout=0.6).to(device)  # lstm
        self.fc = nn.Linear(self.hidden_dim, vocab_size).to(device)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Text Generation")

    parser.add_argument("--epochs", dest="num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", dest="embedding_dim", type=int, default=256)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--window", dest="window", type=int, default=20)
    parser.add_argument("--load_model", dest="load_model", type=bool, default=True)
    parser.add_argument("--model", dest="model", type=str, default='weights/textGenerator_model_10.pt')
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=3)

    return parser.parse_args()


import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def predict(dataset, model, text, next_words=100):
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        x = x.to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        word = dataset.index_to_word[word_index]
        words.append(word)

        if word in {"?", ".", "!"}:
            break

    return words


class Execution:

    def __init__(self, args):
        self.file = 'data/single_cconj.txt'
        self.window = args.window
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.num_layers = args.num_layers

        self.targets = None
        self.sequences = None
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None

    def train(self, args):

        dataset = MyDataset(args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

        # Criterion
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model initialization
        model = TextGenerator(args, len(dataset.uniq_words))
        # Optimizer initialization
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        # Set model in training mode

        model.to(device)

        # Training pahse
        for epoch in range(self.num_epochs):
            model.train()
            state_h, state_c = model.init_state(args.batch_size)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            # Mini batches
            for batch, (x, y) in enumerate(dataloader):
                # Clean gradients

                if batch % 500 == 0:
                    print(batch/len(dataloader))

                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)
                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                loss = criterion(y_pred.transpose(1, 2), y)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch},  loss: {loss.item()} ")
            print(" ".join(predict(dataset, model, text='a')))
            print(" ".join(predict(dataset, model, text='i')))
            print(" ".join(predict(dataset, model, text='the')))
            print(" ".join(predict(dataset, model, text='and')))

            torch.save(model.state_dict(), f'weights/textGenerator_model_{epoch}.pt')

    @staticmethod
    def generator(model, sequences, idx_to_char, n_chars):

        # Set the model in evalulation mode
        model.eval()

        # Define the softmax function
        softmax = nn.Softmax(dim=1)

        # Randomly is selected the index from the set of sequences
        start = np.random.randint(0, len(sequences) - 1)

        # The pattern is defined given the random idx
        pattern = sequences[start]

        # By making use of the dictionaries, it is printed the pattern
        print("\nPattern: \n")
        print(''.join([idx_to_char[value] for value in pattern]), "\"")

        # In full_prediction we will save the complete prediction
        full_prediction = pattern.copy()

        # The prediction starts, it is going to be predicted a given
        # number of characters
        for i in range(n_chars):
            # The numpy patterns is transformed into a tesor-type and reshaped
            pattern = torch.from_numpy(pattern).type(torch.LongTensor)
            pattern = pattern.view(1, -1)

            # Make a prediction given the pattern
            prediction = model(pattern)
            # It is applied the softmax function to the predicted tensor
            prediction = softmax(prediction)

            # The prediction tensor is transformed into a numpy array
            prediction = prediction.squeeze().detach().numpy()
            # It is taken the idx with the highest probability
            arg_max = np.argmax(prediction)

            # The current pattern tensor is transformed into numpy array
            pattern = pattern.squeeze().detach().numpy()
            # The window is sliced 1 character to the right
            pattern = pattern[1:]
            # The new pattern is composed by the "old" pattern + the predicted character
            pattern = np.append(pattern, arg_max)

            # The full prediction is saved
            full_prediction = np.append(full_prediction, arg_max)

        print("Prediction: \n")
        print(' '.join([idx_to_char[value] for value in full_prediction]), "\"")


if __name__ == '__main__':

    args = parameter_parser()

    # If you already have the trained weights
    if args.load_model == True:
        if os.path.exists(args.model):
            dataset = MyDataset(args)

            # Initialize the model
            model = TextGenerator(args, len(dataset.uniq_words))
            # Load weights
            model.load_state_dict(torch.load('weights/textGenerator_model_1.pt'))
            
            print(" ".join(predict(dataset, model, text='a')))
            print(" ".join(predict(dataset, model, text='i')))
            print(" ".join(predict(dataset, model, text='the')))
            print(" ".join(predict(dataset, model, text='and')))

    # If you will train the model
    else:
        # Load and preprare the sequences
        execution = Execution(args)

        print('training')
        # Training the model
        execution.train(args)

        sequences = execution.sequences
        idx_to_char = execution.idx_to_char
        vocab_size = execution.vocab_size

        # Initialize the model
        model = TextGenerator(args, vocab_size)
        # Load weights
        model.load_state_dict(torch.load('weights/textGenerator_model.pt'))

        # Text generator
        execution.generator(model, sequences, idx_to_char, 1000)
