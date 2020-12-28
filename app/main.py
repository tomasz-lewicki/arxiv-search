import os
import pickle

import flask
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

local_dir = os.path.dirname(os.path.realpath(__file__))

with open("pickles/full_cv.pkl", "rb") as f:
    df = pickle.load(f)

with open("pickles/vectorizer.pkl", "rb") as f:
    tfidf_v = pickle.load(f)

with open("pickles/X_tfidf.pkl", "rb") as f:
    X_tfidf = pickle.load(f)


def inference(abstract, top_k=5):

    x = tfidf_v.transform([abstract])
    sims = cos_sim(X_tfidf, x)
    sims = np.squeeze(sims)
    most_sim_idx = sims.argsort()[::-1]  # reverse
    # if we also want to reject the paper itself, then it'd be most_sim_idx[-2::-1]

    df_fajne = df.iloc[most_sim_idx[:top_k]]

    return df_fajne


app = Flask(__name__)

# favicon route
@app.route("/favicon.svg")
def favicon():
    # return send_from_directory(os.path.join(, 'static'),'favicon.svg')
    return send_from_directory(app.root_path + "/static", "favicon.svg")


@app.route("/")
def main_handler():
    return render_template("main.html")


@app.route("/recommend", methods=["POST"])
def rec_handler():

    abstract = request.get_json()["abstract"]
    # abstract = request.form['abstract']

    df = inference(abstract)

    if len(df) < 1:
        return "No papers found.", 404
    else:
        r = {
            "ids": df["id"].tolist(),
            "titles": df["title"].tolist(),
            "authors": df["authors"].tolist(),
            "abstracts": df["abstract"].tolist(),
        }

        return flask.json.jsonify(r), 200


if __name__ == "__main__":

    app.run(debug=True)
