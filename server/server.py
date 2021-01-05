from flask import Flask, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

extra_words=list(STOP_WORDS)+list(punctuation)+['\n']
nlp=spacy.load('en_core_web_sm')



label_dict = {
    0: "More Sad",
    1: "Sad",
    2: "Meh",
    3: "Happy",
    4: "More Happy"
}


@app.route("/run_forward", methods=["POST"])
def forward():
    params = request.get_json()
    sentence = params["sentence"]
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased");
    model = DistilBertForSequenceClassification.from_pretrained("../models/amazon-distilbert")

    tokens = tokenizer(sentence, return_tensors="pt")

    output = model(tokens["input_ids"], tokens["attention_mask"]).logits

    probs = torch.softmax(output, dim=1).tolist()[0]

    label = np.argmax(np.array(probs))

    ret = label_dict[label]

    return {"data": ret}


@app.route("/run_forward_paraphrase", methods=["POST"])
def run_forward_paraphrase():

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    sentence = "This is something which i cannot understand at all"
    text = "paraphrase: " + sentence
    encoding = tokenizer(text, padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )

    ans = ""
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        ans += line

    return {"data": ans}

@app.route("/run_forward_summarizer", methods=["POST"])
def run_forward_summarizer():
    params = request.get_json()
    sentence = params["sentence"]
    docx = nlp(sentence)

    all_words = [word.text for word in docx]
    Freq_word = {}
    for w in all_words:
        w1 = w.lower()
        if w1 not in extra_words and w1.isalpha():
            if w1 in Freq_word.keys():
                Freq_word[w1] += 1
            else:
                Freq_word[w1] = 1

    val = sorted(Freq_word.values())
    max_freq = val[-3:]

    for word in Freq_word.keys():
        Freq_word[word] = (Freq_word[word] / max_freq[-1])

    sent_strength = {}
    for sent in docx.sents:
        for word in sent:

            if word.text.lower() in Freq_word.keys():

                if sent in sent_strength.keys():
                    sent_strength[sent] += Freq_word[word.text.lower()]
                else:

                    sent_strength[sent] = Freq_word[word.text.lower()]

            else:
                continue
    top_sentences = (sorted(sent_strength.values())[::-1])
    top30percent_sentence = int(0.3 * len(top_sentences))

    top_sent = top_sentences[:top30percent_sentence]

    summary = []
    for sent, strength in sent_strength.items():
        if strength in top_sent:
            summary.append(sent)

        else:
            continue
    ans=""
    for i in summary:
        ans += i.text + " "

    return {"data": ans}


if __name__ == "__main__":
    app.run()