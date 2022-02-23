from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy.random as rd
import json


# NN to check, for a pair of numbers, which one is greater.

# ===============================
# Input variables:   A, B
# Expected output:   1 if A > B ? 0
# ===============================


# 1. Data preprocessing
def generate_sample_data(n_sample, name='') -> None:
    # Generate sample data and save it as a json file
    with open(f'data/{name}_inp.json', 'w') as f:
        inp = {i: rd.random(size=2).tolist() for i in range(n_sample)}
        json.dump(inp, f)
    
    with open(f'data/{name}_out.json', 'w') as f:
        json.dump({key: int(val[0] > val[1]) for key, val in inp.items()}, f)


def get_data(name='') -> list:
    # Get processed data
    with open(f'data/{name}_inp.json') as inp:
        inp = list(json.load(inp).values())
    with open(f'data/{name}_out.json') as out:
        out = list(json.load(out).values())
    return inp, out


# 2. NN model
def create_model():
    # Create NN model
    model = Sequential()
    model.add(Dense(units=1, activation='sigmoid', input_dim=2))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
    return model


# 3. Train and test model
def train_model(model):
    # Train model using training data
    inp_training, out_training = get_data(name='train')
    model.fit(inp_training, out_training, epochs=200, batch_size=300)


def test_model(model):
    # Test model using test data
    inp_test, out_test = get_data(name='test')
    model.evaluate(inp_test, out_test)


# ===============================


def main():
    N_TRAINING = 10000
    N_TEST     = 10000
    
    generate_sample_data(N_TRAINING, 'train')
    generate_sample_data(N_TEST, 'test')
    
    model = create_model()
    
    train_model(model)
    test_model(model)
    

if __name__ == '__main__':
    main()