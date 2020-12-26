from flask import Flask, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased");
model = DistilBertForSequenceClassification.from_pretrained("../models/amazon-distilbert")

label_dict = {
    0: "Sad Af",
    1: "Sad",
    2: "Meh",
    3: "Happy",
    4: "Happy af"
}


@app.route("/run_forward", methods=["POST"])
def forward():
    params = request.get_json()
    sentence = params["sentence"]
    print(sentence)

    tokens = tokenizer(sentence, return_tensors="pt")

    output = model(tokens["input_ids"], tokens["attention_mask"]).logits

    probs = torch.softmax(output, dim=1).tolist()[0]

    label = np.argmax(np.array(probs))

    ret = label_dict[label]
    return {"data": ret}


if __name__ == "__main__":
    app.run()
