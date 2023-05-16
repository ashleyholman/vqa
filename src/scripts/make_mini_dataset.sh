#!/bin/bash -e

# Change directory to the data dir
cd $( dirname -- "$0"; )/../../data/
echo Data dir: $(pwd)

if [ ! -f v2_OpenEnded_mscoco_train2014_questions.json ] || [ ! -f v2_mscoco_train2014_annotations.json ]; then
  echo "Training dataset not found.  Make sure you've ran src/scripts/fetch_datasets.py to download it first."
  exit 1
fi

echo Extracting the first 16 questions to subset_questions.json...
jq 'with_entries(if .key == "questions" then .value = .value[0:16] else . end)' v2_OpenEnded_mscoco_train2014_questions.json > subset_questions.json
echo Extracting the associated annotations to subset_annotations.json...
jq '[.questions[].question_id | tostring]' subset_questions.json > question_ids.json
jq --slurpfile ids question_ids.json '
  ($ids[0] | map({(.): .}) | add) as $id_dict
  | with_entries(if .key == "annotations" then .value = [ .value[] | select( .question_id | tostring | in($id_dict) ) ] else . end)' v2_mscoco_train2014_annotations.json > subset_annotations.json
rm -f question_ids.json
