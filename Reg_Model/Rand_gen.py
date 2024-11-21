import tensorflow as tf
import numpy as np

# Function to create the original TFRecord dataset
def create_original_tfrecord(filename, num_records=1000, file_names=None):
    if file_names is None:
        file_names = [f'image_{i}.jpg' for i in range(num_records)]

    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(num_records):
            example = tf.train.Example(features=tf.train.Features(feature={
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_names[i].encode()])),
                'aesthetics_score': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.random.randint(0, 6)])),
                'artifact_score': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.random.randint(0, 6)])),
                'misalignment_score': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.random.randint(0, 6)])),
                'overall_score': tf.train.Feature(int64_list=tf.train.Int64List(value=[np.random.randint(0, 6)]))
            }))
            writer.write(example.SerializeToString())
    print(f"TFRecord file '{filename}' created with {num_records} records.")

# Function to create the self-attention TFRecord dataset
def create_attention_tfrecord(filename, num_records=1000, file_names=None):
    if file_names is None:
        file_names = [f'image_{i}.jpg' for i in range(num_records)]

    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(num_records):
            example = tf.train.Example(features=tf.train.Features(feature={
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_names[i].encode()])),
                'self_attention': tf.train.Feature(float_list=tf.train.FloatList(value=np.random.rand(1024 * 2048).tolist()))
            }))
            writer.write(example.SerializeToString())
    print(f"TFRecord file '{filename}' created with {num_records} records.")

# Function to parse and display the first five rows of a TFRecord file
def print_tfrecord_first_rows(filename, num_rows=5, is_attention=False):
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'aesthetics_score': tf.io.FixedLenFeature([], tf.int64),
        'artifact_score': tf.io.FixedLenFeature([], tf.int64),
        'misalignment_score': tf.io.FixedLenFeature([], tf.int64),
        'overall_score': tf.io.FixedLenFeature([], tf.int64)
    } if not is_attention else {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'self_attention': tf.io.FixedLenFeature([1024 * 2048], tf.float32)
    }

    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))

    print(f"First {num_rows} rows of '{filename}':")
    for i, record in enumerate(parsed_dataset.take(num_rows)):
        parsed_record = {key: record[key].numpy() for key in record}
        print(parsed_record)
        print("-" * 80)

# Create consistent filenames for both datasets
num_records = 1000
file_names = [f'image_{i}.jpg' for i in range(num_records)]

# Create both TFRecord files
create_original_tfrecord("Rand1K.tfrecord", num_records=num_records, file_names=file_names)
create_attention_tfrecord("Rand_atten1K.tfrecord", num_records=num_records, file_names=file_names)

# Print the first 5 rows of each file
print_tfrecord_first_rows("Rand1K.tfrecord", num_rows=5, is_attention=False)
print_tfrecord_first_rows("Rand_atten1K.tfrecord", num_rows=5, is_attention=True)
