from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import random
import numpy as np

class GlobalModel:

    def __init__(self):
        self.tsaModel = DistilBertForSequenceClassification.from_pretrained("/home/vamsi/Documents/GitHub/TextSentimentAnalysis/models/amazon-distilbert")
        self.tsaTokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    def tsaForward(self, sentence):
        tokens = self.tsaTokenizer(sentence, return_tensors="pt")

        output = self.tsaModel(tokens["input_ids"], tokens["attention_mask"]).logits

        probs = torch.softmax(output, dim=1).tolist()[0]

        label = np.argmax(np.array(probs))

        return label



    def sumForward(self):

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
        ans = ""
        for i in summary:
            ans += i.text + " "
        pass

    def paraForward(self):
        pass