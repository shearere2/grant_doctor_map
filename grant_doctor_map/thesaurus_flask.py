import fasttext
import flask
import gensim.models._fasttext_bin
import logging
import numpy as np

# Initialize app
app = flask.Flask('API')
# Load model
fasttext_model_path = 'data/cc.en.50.bin'
model = gensim.models.fasttext.load_facebook_vectors(fasttext_model_path)

@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return flask.jsonify({'success': True})

@app.route('/correction', methods=['GET'])
def correction():
    word = flask.request.args.get('word').lower()
    similar = model.get_nearest_neighbors(model.get_word_vector(word), k=10)
    # Returns [('banana', 0.9), ('potato', 0.3)]
    return flask.jsonify({'success': True, 'similar': similar})

@app.route('/math', methods=['GET'])
def math():
    pos = flask.request.args.get('positive')
    neg = flask.request.args.get('negative')
    if (pos and len(pos) > 0):
        pos = [v.lower() for v in pos.split(',')]
        if (neg and len(neg) > 0):
            neg = [v.lower() for v in neg.split(',')]
        else:
            neg = []
        similar = model.most_similar(positive=pos, negative=neg)
        # Returns [('banana', 0.9), ('potato', 0.3)]
        return flask.jsonify({'success': True, 'similar': similar})

@app.route('/sentence', methods=['GET'])
def sentence():
    sentence = flask.request.args.get('sentence')
    if (sentence and len(sentence) > 0):
        sentence = [v.lower() for v in sentence.split('_')]
        vecs = [model.get_vector(v) for v in sentence]
        vec = np.mean(vecs, axis=0)
        similar = model.similar_by_vector(vec, restrict_vocab=30_000)
        # Returns [('banana', 0.9), ('potato', 0.3)]
        return flask.jsonify({'success': True, 'similar': similar})

@app.route('/etim', methods=['GET'])
def etim():
    return flask.render_template('etim.html')

logging.basicConfig(level=logging.INFO)
app.run(port=8000)