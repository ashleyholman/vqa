import json
import inflect
import os
from transformers import BertModel, BertTokenizer, ViTModel, ViTImageProcessor
import torch
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from functools import lru_cache

from src.models.model_configuration import ModelConfiguration
from src.data.embeddings_manager import EmbeddingsManager

class VQADataset(Dataset):
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

    def __init__(self, config : ModelConfiguration, settype='train', num_dataloader_workers=1, answer_classes_and_substitutions=([], []), with_input_ids=False, with_images_features=False):
        self.images = []
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.question_ids = []
        self.image_ids = []
        self.images = {}
        self.settype = settype
        self.question_embeddings = []
        self.question_texts = []
        self.image_embeddings = []
        self.image_paths = []
        self.image_count = 0
        self.with_input_ids = with_input_ids
        self.with_images_features = with_images_features
        self.config = config
        self.embeddings_manager = EmbeddingsManager(config, settype, num_dataloader_workers)

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL_NAME)
        self.vit_preprocessor = ViTImageProcessor.from_pretrained(self.VIT_MODEL_NAME)

        if settype == 'train':
          annotations_json_file = self.TRAIN_ANNOTATIONS_JSON_FILE_NAME
          questions_json_file = self.TRAIN_QUESTIONS_JSON_FILE_NAME
          self.image_prefix = self.TRAIN_IMAGE_PREFIX
        elif settype == 'validation':
          annotations_json_file = self.VALIDATION_ANNOTATIONS_JSON_FILE_NAME
          questions_json_file = self.VALIDATION_QUESTIONS_JSON_FILE_NAME
          self.image_prefix = self.VALIDATION_IMAGE_PREFIX
        elif settype == 'mini':
          annotations_json_file = self.MINI_ANNOTATIONS_JSON_FILE_NAME
          questions_json_file = self.MINI_QUESTIONS_JSON_FILE_NAME
          self.image_prefix = self.MINI_IMAGE_PREFIX
        else:
          raise ValueError("Unknown set type: " + settype)

        # Load the questions and annotations json files
        with open(annotations_json_file) as f:
            annotations = json.load(f)['annotations']
        with open(questions_json_file) as f:
            questions = json.load(f)['questions']

        # Create a reverse mapping from question_id to annotation
        question_to_annotation = {a['question_id']: a for a in annotations}

        if not all(answer_classes_and_substitutions):
            # Caller did not specify answer_classes, so build them from the most common answers
            print("Building answer classes from the most common answers...")
            self.answer_classes, self.answer_substitutions = VQADataset.build_answer_classes(annotations)
        else:
            self.answer_classes = answer_classes_and_substitutions[0]
            self.answer_substitutions = answer_classes_and_substitutions[1]
        # apply the answer classes as answer_class_id's to the annotations set
        print("Applying answer classes to annotations...")
        VQADataset.apply_answer_classes(annotations, self.answer_classes, self.answer_substitutions)

        # Build a count of the number of answers per class.  This can be used to analyse class imbalance.
        self.class_counts = self._count_classes(annotations, len(self.answer_classes))

        # Tokenizing the question text into input_ids takes time.  Only do it if
        # the user requested input_ids to be in the dataset by setting
        for question in questions:
            if self.with_input_ids:
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
            self.image_paths.append(self.image_prefix + str(question['image_id']).zfill(12) + '.jpg')
            self.question_ids.append(question['question_id'])
            self.question_texts.append(question['question'])

        # Load pre-processed embeddings for question texts and images.  EmbeddingsManager will
        # generate and store them if they don't already exist.
        self.question_embeddings = self.embeddings_manager.get_embeddings('text', self.question_texts)
        self.image_embeddings = self.embeddings_manager.get_embeddings('vision', self.image_paths)

        print("Done initialising dataset")

    # Our dataset has many answers per question, written by different humans.
    # But there's also an attribute called "multiple_choice_answer" which is the most common answer.
    # To make our model simpler we will predict answer classes instead of answering in natural language.
    # Rather than create an answer class for every unique multiple_choice_answer, we'll take the top 999
    # most common answers and make them labels, and then add an "other" label that all other answers will be.
    # This will leave us with 1000 answer classes to predict on.
    @staticmethod
    def build_answer_classes(annotations, merge_singular_plural_classes=True):
        p = inflect.engine()
        answer_counts = Counter(a['multiple_choice_answer'] for a in annotations)
        final_counts = {}
        substitutions = {}

        if merge_singular_plural_classes:
            for answer, count in answer_counts.items():
                # exclude problem cases where either inflect is mistakingly
                # considering it a plural when it's not, or we don't want to
                # merge it with its plural since they can have different meanings in
                # some contexts.
                if answer in ['brass', 'bus', 'cs', 'doubles', 'dominos', 'downs' 'fries', 'glasses',
                              'lots', 'shades', 'singles', 'ss', 'trunks', 'us', 'uss']:
                    continue
                singular = p.singular_noun(answer)
                if (singular and singular != answer and singular in answer_counts):
                    # This condition means that the current answer is a plural
                    # (otherwise singular would be False) and there exists a
                    # singular form of this answer also in the answer_counts list.
                    plural = answer

                    if singular in substitutions or plural in substitutions:
                        # Not expected to happen. Avoid chaining multiple
                        # substituions for any reason.
                        print(f"WARNING: Skip substituting ({singular} / {plural}) to avoid double substitutions")
                        continue

                    if (answer_counts[singular] > answer_counts[plural]):
                        # The singular form is more common, so use that as the
                        # answer class
                        class_to_consolidate = plural
                        class_to_keep = singular
                    else:
                        # The plural form is more common, so use that as the
                        # answer class
                        class_to_consolidate = singular
                        class_to_keep = plural

                    # Record the substitutions
                    substitutions[class_to_consolidate] = class_to_keep

            # Now that we have a list of substitutions, consolidate the counts into final_counts
            for answer, count in answer_counts.items():
                if answer in substitutions:
                    key_to_increment = substitutions[answer]
                else:
                    key_to_increment = answer

                final_counts[key_to_increment] = final_counts.get(key_to_increment, 0) + count
        else:
            final_counts = answer_counts

        # Sort final_counts and keep the top 999 answers
        top_answers = [answer for answer, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:999]]

        # Add 'other' class
        top_answers.append('other')

        # the substitutions list can be reduced to only contain substituions that map to a top answer
        substitutions = {k: v for k, v in substitutions.items() if v in top_answers}

        return top_answers, substitutions

    def apply_answer_classes(annotations, answer_classes, substitutions):
        # Map each answer to its class
        for annotation in annotations:
            answer = annotation['multiple_choice_answer']
            if answer in substitutions:
                # apply substitution
                answer = substitutions[answer]
            if answer not in answer_classes:
                # assign a class value of 999 which represents the "other" class
                annotation['answer_class_id'] = 999
            else:
                annotation['answer_class_id'] = answer_classes.index(answer)

    def _count_classes(self, annotations, num_classes):
        class_counts = np.zeros(num_classes, dtype=int)
        for annotation in annotations:
            class_counts[annotation['answer_class_id']] += 1
        return class_counts

    def __len__(self):
        return max(len(self.input_ids), len(self.question_embeddings))

    def __getitem__(self, idx):
        item = {
            'label': self.labels[idx],
            'question_id': self.question_ids[idx],
            'image_id': self.image_ids[idx],
            'image_embedding': self.image_embeddings[idx],
            'question_embedding': self.question_embeddings[idx]
        }

        if self.with_input_ids:
            item['input_ids'] = self.input_ids[idx]
            item['attention_mask'] = self.attention_masks[idx]

        if self.with_images_features:
            item['image'] = self.preprocess_image(self.image_ids[idx])

        return item