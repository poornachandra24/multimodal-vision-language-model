#!/bin/bash

MODEL_PATH="/home/poorna/pc_projects/multimodal-vision-language-model/models/paligemma-weights/paligemma-3b-pt-224"
PROMPT="
    You are a helpful assistant. 
    Answer the following question based on the image provided.
    Question: What is  in the image?
    Answer:The images shows"
IMAGE_FILE_PATH="/home/poorna/pc_projects/multimodal-vision-language-model/dev/test_image/image2.png"
MAX_TOKENS_TO_GENERATE=1000
TEMPERATURE=0.7
TOP_P=0.8
DO_SAMPLE="True"
ONLY_CPU="True"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU
