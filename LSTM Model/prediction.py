# Importing the Libraries for the Sentiment Analysis project
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# Tokenization of the input text
model = load_model(filepath='model.h5')
twt = ['Meetings: Because none of us is as dumb as all of us.']
max_fatures = 20000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=29, dtype='int32', value=0)
print(twt)

# Prediction
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")