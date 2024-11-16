from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi
import base64
import json
import pickle 
import tensorflow as tf
from datasets import load_dataset
import os.path
from tqdm import tqdm

def parse_tfrecord(record):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    return example

def build_lookup_table(pickapic, split):
    lookup_table = {}
    for item in tqdm(pickapic[split], desc=f"Building lookup table for {split}"):
        lookup_table[item['image_0_uid']] = item
        lookup_table[item['image_1_uid']] = item
    return lookup_table

def get_caption_and_images(lookup_table, filename):
    results = []
    if filename:
        uid = filename.split('/')[-1].split('.')[0]
        if uid in lookup_table:
            matching_item = lookup_table[uid]
            if matching_item['image_0_uid'] == uid:
                caption = matching_item.get('caption', '')
                image = matching_item.get('jpg_0', '')
            elif matching_item['image_1_uid'] == uid:
                caption = matching_item.get('caption', '')
                image = matching_item.get('jpg_1', '')
            results.append({
                'filename': uid, 
                'caption': caption, 
                'jpg': image, 
            })
    return results

def read_tfrecord_file(file_path, split='train', pickapic=None):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    i = 0
    now_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir_path = os.path.join(now_dir, split)
    lookup_table = build_lookup_table(pickapic, split)

    processed_data = list()
    for raw_record in tqdm(raw_dataset):
        save_path = os.path.join(target_dir_path, str(i))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        example = parse_tfrecord(raw_record)
        record = dict()
        record_images = dict()
        for key, value in example.features.feature.items():
            if value.bytes_list.value:
                try:
                    record[key] = value.bytes_list.value[0].decode('utf-8')
                except UnicodeDecodeError:
                    with open(save_path + '/' + str(key)+'.jpg', 'wb') as f:
                        f.write(value.bytes_list.value[0])
                    record[key] = str(key)+'.jpg'
                    record_images[key] = value.bytes_list.value[0]
            elif value.float_list.value:
                record[key] = value.float_list.value[0]
            elif value.int64_list.value:
                record[key] = value.int64_list.value[0]

        results_richhf = get_caption_and_images(lookup_table, record['filename'])
        image = results_richhf
        with open(save_path + '/image.jpg', 'wb') as f:
            f.write(results_richhf[0]['jpg'])
            
        record_images['image'] = results_richhf[0]['jpg']
        record['caption'] = results_richhf[0]['caption']
        json_records = json.dumps(record, indent=4)
        with open(save_path + '/output.json', 'w') as json_file:
            json_file.write(json_records)
        i += 1
        processed_data.append(record | record_images)
    
    return processed_data

print("Loading pickapic dataset...")
pickapic = load_dataset("yuvalkirstain/pickapic_v1", num_proc=64)

file_path = "./richhf-18k/train.tfrecord" 
train_records = read_tfrecord_file(file_path, split='train', pickapic=pickapic)

file_path = "./richhf-18k/test.tfrecord" 
test_records = read_tfrecord_file(file_path, split='test', pickapic=pickapic)

file_path = "./richhf-18k/dev.tfrecord" 
dev_records = read_tfrecord_file(file_path, split='train', pickapic=pickapic)

train_dataset = Dataset.from_list(train_records)
test_dataset = Dataset.from_list(test_records)
dev_dataset = Dataset.from_list(dev_records)

full_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
    "dev": dev_dataset
})

full_dataset.save_to_disk('./rich_human_feedback_dataset')
