# Importing the Libraries for the Sentiment Analysis project
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# Tokenization of the input text and loading the sentiment model
model = load_model(filepath='model.h5')
twt = ['I am happy.']
tokenizer = Tokenizer(num_words=10, split= ' ')
tokenizer.fit_on_texts(twt)
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=29, dtype='int32', value=0)

# Prediction
sentiment = model.predict(twt,batch_size=1,verbose = 2)
print(sentiment)
if(np.argmax(sentiment[0]) == 0):
    print("negative")
elif (np.argmax(sentiment[0]) == 1):
    print("positive")