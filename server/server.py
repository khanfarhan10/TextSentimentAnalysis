from flask import Flask, request

app = Flask(__name__)


@app.route("/run_forward", methods=["POST"])
def forward():
    params = request.get_json()
    sentence = params["sentence"]
    print(sentence)
    return {"data": sentence}


if __name__ == "__main__":
    app.run()