import json
import os
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
import torch
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from functools import lru_cache

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')

VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"  # name of the ViT model
BERT_MODEL_NAME = "bert-base-uncased"

# Training data set
TRAIN_ANNOTATIONS_JSON_FILE_NAME = 'data/v2_mscoco_train2014_annotations.json'
TRAIN_QUESTIONS_JSON_FILE_NAME = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
TRAIN_IMAGE_PREFIX = 'data/train2014/COCO_train2014_'

# Validation data set
VALIDATION_ANNOTATIONS_JSON_FILE_NAME = 'data/v2_mscoco_val2014_annotations.json'
VALIDATION_QUESTIONS_JSON_FILE_NAME = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
VALIDATION_IMAGE_PREFIX = 'data/val2014/COCO_val2014_'

# Mini data set (tiny subset of training data, for testing code locally)
MINI_ANNOTATIONS_JSON_FILE_NAME = 'data/subset_annotations.json'
MINI_QUESTIONS_JSON_FILE_NAME = 'data/subset_questions.json'
MINI_IMAGE_PREFIX = 'data/train2014/COCO_train2014_'

class VQADataset(Dataset):
    def __init__(self, settype='train', answer_classes=[]):
        self.images = []
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.image_ids = []
        self.images = {}
        self.settype = settype

        self.image_count = 0

        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.vit_preprocessor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

        if settype == 'train':
          annotations_json_file = TRAIN_ANNOTATIONS_JSON_FILE_NAME
          questions_json_file = TRAIN_QUESTIONS_JSON_FILE_NAME
          self.image_prefix = TRAIN_IMAGE_PREFIX
        elif settype == 'validation':
          annotations_json_file = VALIDATION_ANNOTATIONS_JSON_FILE_NAME
          questions_json_file = VALIDATION_QUESTIONS_JSON_FILE_NAME
          self.image_prefix = VALIDATION_IMAGE_PREFIX
        elif settype == 'mini':
          annotations_json_file = MINI_ANNOTATIONS_JSON_FILE_NAME
          questions_json_file = MINI_QUESTIONS_JSON_FILE_NAME
          self.image_prefix = MINI_IMAGE_PREFIX
        else:
          raise ValueError("Unknown set type: " + settype)

        # Load the questions and annotations json files
        with open(annotations_json_file) as f:
            annotations = json.load(f)['annotations']
        with open(questions_json_file) as f:
            questions = json.load(f)['questions']

        # Create a reverse mapping from question_id to annotation
        question_to_annotation = {a['question_id']: a for a in annotations}

        if not answer_classes:
            # Caller did not specify answer_classes, so build them from the most common answers
            print("Building answer classes from the most common answers...")
            self.answer_classes = VQADataset.build_answer_classes(annotations)
        else:
            self.answer_classes = answer_classes
        # apply the answer classes as answer_class_id's to the annotations set
        print("Applying answer classes to annotations...")
        VQADataset.apply_answer_classes(annotations, self.answer_classes)
        #print(f"Answer classes: {self.answer_classes}")

        # Build a count of the number of answers per class.  This can be used to analyse class imbalance.
        self.class_counts = self._count_classes(annotations, len(self.answer_classes))

        # Our dataset will have one sample per question, each question will have one image and one answer class.
        # So, iterate over the questions to preprocess the dataset one sample at a time
        print("Tokenizing question text...")
        for question in questions:
            encoding = self.bert_tokenizer.encode_plus(
                question['question'], 
                max_length=30, 
                truncation=True, 
                padding='max_length', 
                return_tensors='pt'
            )
            self.input_ids.append(encoding['input_ids'].flatten())
            self.attention_masks.append(encoding['attention_mask'].flatten())
            self.labels.append(question_to_annotation[question['question_id']]['answer_class_id'])
            self.image_ids.append(question['image_id'])
            
            # Pre-process the image if it hasn't been yet
            #image_id = question['image_id']
            #if self.images.get(image_id) is None:
            #    self.preprocess_image(image_id)
        print("Done initialising dataset")

    # Our dataset has many answers per question, written by different humans.
    # But there's also an attribute called "multiple_choice_answer" which is the most common answer.
    # To make our model simpler we will predict answer classes instead of answering in natural language.
    # Rather than create an answer class for every unique multiple_choice_answer, we'll take the top 999
    # most common answers and make them labels, and then add an "other" label that all other answers will be.
    # This will leave us with 1000 answer classes to predict on.
    @staticmethod
    def build_answer_classes(annotations):
        # Identify top 999 answer classes
        answer_counts = Counter(a['multiple_choice_answer'] for a in annotations)
        answer_classes = [answer for answer, count in answer_counts.most_common(999)]

        # Add 'other' class
        answer_classes.append('other')

        return answer_classes

    def apply_answer_classes(annotations, answer_classes):
        # Map each answer to its class
        for annotation in annotations:
            answer = annotation['multiple_choice_answer']
            if answer not in answer_classes:
                # assign a class value of 999 which represents the "other" class
                annotation['answer_class_id'] = 999
                #print(f"Assigning answer \"{answer}\" to 'other'")
            else:
                annotation['answer_class_id'] = answer_classes.index(answer)
                #print(f"Answer \"{answer}\" is in the top 999.. assigning with index {top_answers.index(answer)}")

    def _count_classes(self, annotations, num_classes):
        class_counts = np.zeros(num_classes, dtype=int)
        for annotation in annotations:
            class_counts[annotation['answer_class_id']] += 1
        return class_counts


    @lru_cache(maxsize=1000)
    def preprocess_image(self, image_id):
        image_path = self.image_prefix + str(image_id).zfill(12) + '.jpg'
        #print(f"CACHE MISS on {image_path}")

        image = Image.open(image_path)
        image = image.convert("RGB")

        # Convert the image to a numpy array and pass it to the feature extractor
        image = np.array(image)
        features = self.vit_preprocessor(image, return_tensors='pt')

        # return the pixel_values features
        return features['pixel_values'].squeeze(0)

    # This method is used to pre-compute the embeddings for all images and question text in the dataset.
    # This is so that we can feed the embeddings directly into our model at training/validation time,
    # rather than having to compute them on the fly.  This will *hopefully* provide a significant speedup
    # for training and validation.
    def generate_and_save_all_embeddings(self, num_dataloader_workers=1):
        BATCH_SIZE = 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # We need both images and questions for this operation
        data_dataloader = DataLoader(self, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_dataloader_workers)

        # Move both models to GPU if available
        img_model = ViTModel.from_pretrained(VIT_MODEL_NAME).to(device)
        txt_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)

        all_img_embeddings = []
        all_txt_embeddings = []

        print('Generating embeddings...')

        for idx, batch in enumerate(data_dataloader, start=1):
            images = batch['image'].to(device)  # move images to device
            input_ids = batch['input_ids'].to(device)  # move input_ids to device
            attention_masks = batch['attention_mask'].to(device)  # move attention_mask to device

            with torch.no_grad():
                img_embeddings = img_model(images)['pooler_output']
                txt_embeddings = txt_model(input_ids, attention_mask=attention_masks)['pooler_output']

            # Print the first 5 features of each tensor.
            print('Image embeddings:\n', img_embeddings[:, :5])  # Takes all rows, but only first 5 columns
            print('Text embeddings:\n', txt_embeddings[:, :5])  # Takes all rows, but only first 5 columns

            all_img_embeddings.append(img_embeddings.cpu())
            all_txt_embeddings.append(txt_embeddings.cpu())

            if idx % 1 == 0:
                this_batch_size = batch['image'].shape[0]
                print(f'Processed {((idx-1) * BATCH_SIZE)+this_batch_size}/{len(self)} images and questions...')

        all_img_embeddings = torch.cat(all_img_embeddings)
        all_txt_embeddings = torch.cat(all_txt_embeddings)
        print(f'Finished processing {len(all_img_embeddings)} images and questions...')

        save_path = os.path.join(DATA_DIR, f'{self.settype}_embeddings.pt')

        torch.save({
            'img_embeddings': all_img_embeddings,
            'txt_embeddings': all_txt_embeddings
        }, save_path)

        print(f'All embeddings saved to {save_path}.')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.preprocess_image(image_id)

        return {
            'image': image,
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx]
        }