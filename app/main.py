import os
import pickle
import time 

import flask
import scipy
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

# For fast column lookups
X_tfidf_csc = X_tfidf.tocsc()

def calculate_cos_sim_fast(query_str: str):
    # Performs query only on columns existing in the 
    # Time complexity is O(n) where n is # of unique tokens
    # This is more efficinet, but only for small inputs 
    
    # tokenize the string
    x = np.array(tfidf_v.transform([query_str]).todense())
    
    # construct query vector
    x_nonzero_idx = x.nonzero()[1]
    x_small = x[:,x_nonzero_idx]
    
    # Narrow the similarity computation to tokens that are present in the query_string
    X_words_sparse = scipy.sparse.hstack([X_tfidf_csc.getcol(c) for c in x_nonzero_idx])

    # the vectors are already normalized
    # we can thus limit a*b/(norm(a)*norm(b)) to a*b
    cos_sim = X_words_sparse @ x_small.T

    return cos_sim



def inference(abstract, top_k=5):
    st = time.perf_counter()
    similarities = calculate_cos_sim_fast(abstract)
    # if we also want to reject the paper itself, then it'd be most_sim_idx[-2::-1]
    most_sim_idx = similarities.squeeze().argsort()[::-1]
    df_similar = df.iloc[most_sim_idx[:5]]

        
    print(f"{1000 * (time.perf_counter() - st)}")

    return df_similar


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
