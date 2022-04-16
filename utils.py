import torch
import textstat


def sent_scoring(model_tokenizer, text, cuda):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    assert model is not None
    assert tokenizer is not None
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    if cuda:
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    sentence_prob = loss.item()
    return sentence_prob


def get_reading_level(text):

    level = textstat.flesch_reading_ease(text)

    if level > 90:
        return 'very_easy'
    elif level > 80:
        return 'easy'
    elif level > 70:
        return 'fairly_easy'
    elif level > 60:
        return 'standard'
    elif level > 50:
        return 'fairly_difficult'
    elif level > 30:
        return 'difficult'
    else:
        return 'very_confusing'
