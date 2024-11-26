import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import sys

# Function to parse the original dataset
def parse_original_tfrecord(example):
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),  # Filename as a key
        'aesthetics_score': tf.io.FixedLenFeature([], tf.int64),
        'artifact_score': tf.io.FixedLenFeature([], tf.int64),
        'misalignment_score': tf.io.FixedLenFeature([], tf.int64),
        'overall_score': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    return {
        'filename': parsed_example['filename'],
        'labels': tf.stack([
            parsed_example['aesthetics_score'],
            parsed_example['artifact_score'],
            parsed_example['misalignment_score'],
            parsed_example['overall_score']
        ])
    }

# Function to parse the self-attention TFRecord dataset
def parse_attention_tfrecord(example):
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),  # Filename as a key
        'self_attention': tf.io.FixedLenFeature([1024 * 2048], tf.float32)  # Flattened self-attention matrix
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    return {
        'filename': parsed_example['filename'],
        'self_attention': parsed_example['self_attention']
    }

# Load both TFRecord datasets
original_tfrecord_file = "Rand1K.tfrecord"
attention_tfrecord_file = "Rand_atten1K.tfrecord"

original_dataset = tf.data.TFRecordDataset(original_tfrecord_file).map(parse_original_tfrecord)
print(original_dataset)
attention_dataset = tf.data.TFRecordDataset(attention_tfrecord_file).map(parse_attention_tfrecord)

# Convert both datasets to dictionaries keyed by filename
original_data_dict = {item['filename'].numpy().decode(): {
    'labels': item['labels'].numpy()
} for item in original_dataset}

attention_data_dict = {item['filename'].numpy().decode(): item['self_attention'].numpy() for item in attention_dataset}

# Merge datasets by filename
merged_data = []
for filename, original_data in original_data_dict.items():
    if filename in attention_data_dict:  
        merged_data.append({
            'features': attention_data_dict[filename],
            'labels': original_data['labels'][3]
        })

# Convert merged data to NumPy arrays
X = np.array([item['features'] for item in merged_data])
y = np.array([item['labels'] for item in merged_data])

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6667, random_state=42)  # 10% val, 20% test

model = MLPRegressor()

params = {'hidden_layer_sizes': [10,50,100,200], 'activation': ['logistic', 'tanh', 'relu'], 'solver': ['sgd', 'adam'], 
        'batch_size': [10,50,200], 'learning_rate_init': [0.001, 0.01]}


print("starting grid search", file=sys.stderr)
#print("starting grid search")

grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', verbose=0)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(f"Best Params: {grid.best_params_}")


