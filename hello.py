
from email import message
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from util import data


def model_output(seed_text):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(data)
      model = tf.keras.models.load_model('saved_poetry_generation.h5')
      next_words = 20
      seed_text =str(seed_text)
      for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=134, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
           if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
      
      return seed_text








app = Flask(__name__)

@app.route( '/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input = request.form["text_input"]
        
        return render_template('index.html',out=model_output(input))
    else:
        return render_template('index.html')





if __name__ =='__main__':
    app.run(port=3000,debug = True)
