from flask import Flask, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased");
model = DistilBertForSequenceClassification.from_pretrained("/models/amazon-distilbert", local_files_only=False)

@app.route("/run_forward", methods=["POST"])
def forward():
    params = request.get_json()
    sentence = params["sentence"]
    print(sentence)

    tokens = tokenizer(sentence, return_tensors="pt")

    output = model(tokens["input_ids"], tokens["attention_mask"]).logits

    probs = torch.softmax(output, dim=1).tolist()[0]

    print(probs)

    return {"data": sentence}


if __name__ == "__main__":
    app.run()