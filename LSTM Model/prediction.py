# Importing the Libraries for the Sentiment Analysis project
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# Tokenization of the input text and loading the sentiment model
model = load_model(filepath='model.h5')
text = ["Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"]
text[0] = text[0].lower()
text[0] = re.sub('[^A-Za-z0-9.]+', ' ', text[0])
text[0] = text[0].replace('rt', ' ')
tokenizer = Tokenizer(num_words=100, split= ' ')
tokenizer.fit_on_texts(text)
text = tokenizer.texts_to_sequences(text)
text = pad_sequences(text, maxlen=29, dtype='int32', value=0)
print(text)

# Prediction
sentiment = model.predict(text,batch_size=1,verbose = 2)
print(sentiment)
if(np.argmax(sentiment[0]) == 0):
    print("negative")
elif (np.argmax(sentiment[0]) == 1):
    print("positive")