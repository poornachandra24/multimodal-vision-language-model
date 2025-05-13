#!/bin/bash

MODEL_PATH="/home/poorna/pc_resume_projects/multimodal-vision-language-model/models/paligemma-weights/paligemma-3b-pt-224"
PROMPT="tell me about this animal "
IMAGE_FILE_PATH="/home/poorna/pc_resume_projects/multimodal-vision-language-model/dev/test_image/image.png"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU
