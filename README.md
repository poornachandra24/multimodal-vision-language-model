# Multimodal Vision Language Model from Scratch (PaliGemma Inspired)

This project is an attempt to learn about and build a multimodal vision-language model, drawing inspiration from architectures like Google's PaliGemma. The codebase implements key components of such a model in PyTorch from scratch, including:

*   **Vision Encoder:** A SigLIP-like Vision Transformer (ViT) to process images, using standard self-attention mechanisms.
*   **Language Model:** A Gemma-like decoder-only Transformer for text understanding and generation. This implementation incorporates modern LLM techniques such as:
    *   **Rotary Positional Embeddings (RoPE):** For injecting relative positional information directly into the attention mechanism.
    *   **Grouped-Query Attention (GQA):** An efficient attention variant that groups key and value heads to reduce memory bandwidth during inference, particularly with KV Caching.
    *   RMSNorm for layer normalization.
*   **Multimodal Projection:** A projector to align image and text embeddings.
*   **Input Processing:** Custom processor to prepare image and text inputs according to PaliGemma's formatting.
*   **Inference:** Script to run the model for image-conditioned text generation, including optimized **KV Caching** compatible with GQA.
*   **Experiment Tracking:** Integrated with [Comet ML](https://www.comet.com/) for logging parameters, metrics, and outputs.

**Disclaimer:** This is a learning project. While it aims to replicate components of established models, it's built from scratch for educational purposes and may have differences or simplifications compared to production models.
## Project Structure
```
multimodal-vision-language-model/
├── dev/ # Development and source code
│ ├── get_tokenizer.py # (Script to download model weights and tokenizer)
│ ├── inference.py # Main script for running inference
│ ├── launch_inference.sh # Shell script to launch inference.py
│ ├── modeling_gemma.py # Gemma-like language model implementation
│ ├── modeling_siglip.py # SigLIP-like vision model implementation
│ ├── processing_paligemma.py # Input processor implementation
│ ├── utils.py # Utility functions (e.g., model loading)
│ └── test_image/ # Directory for sample test images
│   └── image2.png # Example test image (replace with your actual image name)
├── models/ # Directory for storing model weights and configs
│ └── paligemma-weights/
│ └── paligemma-3b-pt-224/ # Stores downloaded PaliGemma model files
├── LICENSE
├── README.md
└── requirements.txt # Python dependencies
```
## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd multimodal-vision-language-model
    ```

2.  **Create a Virtual Environment and Install Dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` is up-to-date with packages like `torch`, `transformers`, `Pillow`, `fire`, `safetensors`, `numpy`)*

3.  **Comet ML Setup (Optional, for experiment tracking):**
    *   Sign up for a free account at [Comet.ml](https://www.comet.com/).
    *   Get your API key from your Comet account settings.
    *   Configure Comet ML by setting environment variables or creating a `.comet.config` file. The simplest way is often environment variables:
        ```bash
        export COMET_API_KEY="YOUR_API_KEY"
        export COMET_PROJECT_NAME="multimodal-vision-model" # Or your preferred project name
        export COMET_WORKSPACE="YOUR_WORKSPACE_NAME"
        ```
    *   The inference script will automatically log to Comet if these are set and `comet_ml` is installed.

4.  **Download Model Weights and Tokenizer:**
    The model architecture is designed to be compatible with pre-trained PaliGemma weights. You need to download these from Hugging Face Hub.
        ```bash
        python dev/get_tokenizer.py
        ```

    *   **Hugging Face CLI - Requires Huggingface API Token for gated Models:**
        Make sure you are logged into Hugging Face CLI:
        ```bash
        huggingface-cli login
        ```
    You can run `get_tokenizer.py`

    After running the download script, your `models/paligemma-weights/paligemma-3b-pt-224/` directory should contain:
    *   `config.json`
    *   `tokenizer.json` (or `tokenizer.model`)
    *   `special_tokens_map.json`
    *   `tokenizer_config.json`
    *   Multiple `.safetensors` files (the model weights)

5.  **Prepare a Test Image:**
    Place your test image (e.g., `image2.png` as shown in the example output) in the `dev/test_image/` directory. Update `IMAGE_FILE_PATH` in `dev/launch_inference.sh` if your image has a different name.

## Running Inference

The `dev/launch_inference.sh` script is configured to run inference with default parameters.

1.  **Review and Modify `dev/launch_inference.sh` (Optional):**
    You can edit this script to change:
    *   `MODEL_PATH`: Should point to your downloaded model files.
    *   `PROMPT`: The text prompt to use with the image.
    *   `IMAGE_FILE_PATH`: Path to your test image.
    *   `MAX_TOKENS_TO_GENERATE`, `TEMPERATURE`, `TOP_P`, `DO_SAMPLE`: Generation parameters.
    *   `ONLY_CPU`: Set to `True` to force CPU usage.

2.  **Execute the Launch Script:**
    Make sure you are in the `dev` directory or adjust paths in the script accordingly if running from the project root. If in the project root:
    ```bash
    cd dev
    sh launch_inference.sh
    ```
    Or from the project root:
    ```bash
    sh dev/launch_inference.sh
    ```

This will load the model, process the image and prompt, and generate text.

## Example Output
Below is an example of the model's output for the provided image of a person running, along with some Comet ML logging information.
- **Test Image:**
![alt text](image.png)
- **Console Output & Comet Log Snippet:**
```
User Input:  
    You are a helpful assistant. 
    Answer the following question based on the image provided.
    Question: What is  in the image?
    Answer:The images shows
AI Response:         1. Running
        2. Running shoes
        3. Footwear
        4. Athletic apparel
        5. Athletic shoe
COMET INFO: ---------------------------------------------------------------------------------------
COMET INFO: Comet.ml Experiment Summary
COMET INFO: ---------------------------------------------------------------------------------------
COMET INFO:   Data:
COMET INFO:     display_summary_level : 1
COMET INFO:     name                  : explicit_convertible_6954
COMET INFO:     url                   : https://www.comet.com/poornachandra24/multimodal-vision-model/YOUR_EXPERIMENT_ID
COMET INFO:   Parameters:
COMET INFO:     Decoded Generated tokens            :         1. Running
        2. Running shoes
        3. Footwear
        4. Athletic apparel
        5. Athletic shoe
COMET INFO:     Encoded Generated tokens            : [   145 235274 235265  35574    108    145 235284 235265  35574   9378
    108    145 235304 235265 187829    108    145 235310 235265  54292
  49126    108    145 235308 235265  54292  22043      1]
COMET INFO:     Final Attention mask                : [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
  1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
COMET INFO:     Inital Attention mask               : [[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1]]
COMET INFO:     Input IDs                           : [[257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152 257152 257152 257152 257152
  257152 257152 257152 257152 257152 257152      2    108    141   2045
     708    476  10055  20409 235265 235248    108    141   1261    573
    2412   2872   3482    611    573   2416   4646 235265    108    141
    9413 235292   2439    603    139    473    573   2416 235336    108
     141   1261 235292    651   5191   4918    108]]
COMET INFO:     KV Cache                            : <modeling_gemma.KVCache object at 0x71418371fbe0>
COMET INFO:     Next token after top-p sampling     : 1
COMET INFO:     Next token logits                   : [[-10.106873   10.551077   -5.2431364 ...  -4.8431416  -4.835617
   -4.8543415]]
COMET INFO:     Next token logits after temperature : [[1.7169632e-15 1.1255899e-02 1.7878190e-12 ... 3.1658356e-12
  3.2000513e-12 3.1155831e-12]]
COMET INFO:     Output KV cache                     : {'logits': '[[[-14.717224  13.759519 -10.258333 ...  -8.591268  -8.60586   -8.619645]]]', 'kv_cache': '<modeling_gemma.KVCache object at 0x71418371fbe0>'}
COMET INFO:     Outputs in loop                     : {'logits': '[[[-14.717224  13.759519 -10.258333 ...  -8.591268  -8.60586   -8.619645]]]', 'kv_cache': '<modeling_gemma.KVCache object at 0x71418371fbe0>'}
COMET INFO:     Pixel values                        : [[[[ 0.79607844  0.79607844  0.8039216  ...  0.3176471   0.27058828
     0.27843142]
   [ 0.79607844  0.79607844  0.8039216  ...  0.4039216   0.35686278
     0.36470592]
   [ 0.79607844  0.79607844  0.8039216  ...  0.48235297  0.427451
     0.4431373 ]
   ...
   [-0.30196077 -0.27058822 -0.27843136 ... -0.32549018 -0.30196077
    -0.29411763]
   [-0.2862745  -0.27843136 -0.2862745  ... -0.29411763 -0.27843136
    -0.2862745 ]
   [-0.3098039  -0.3098039  -0.29411763 ... -0.23137254 -0.23137254
    -0.25490195]]

  [[ 0.8117647   0.8117647   0.8117647  ...  0.30980396  0.26274514
     0.27058828]
   [ 0.8117647   0.8117647   0.8117647  ...  0.39607847  0.34901965
     0.35686278]
   [ 0.8117647   0.8117647   0.8117647  ...  0.47450984  0.41960788
     0.43529415]
   ...
   [-0.30196077 -0.27058822 -0.27843136 ... -0.32549018 -0.3098039
    -0.30196077]
   [-0.29411763 -0.2862745  -0.2862745  ... -0.30196077 -0.2862745
    -0.29411763]
   [-0.31764704 -0.31764704 -0.30196077 ... -0.23921567 -0.23921567
    -0.26274508]]

  [[ 0.7490196   0.7490196   0.7490196  ...  0.254902    0.20784318
     0.21568632]
   [ 0.7490196   0.7490196   0.7490196  ...  0.3411765   0.2941177
     0.30196083]
   [ 0.7490196   0.7490196   0.7490196  ...  0.41960788  0.36470592
     0.3803922 ]
   ...
   [-0.372549   -0.34117645 -0.3490196  ... -0.3960784  -0.38039213
    -0.372549  ]
   [-0.36470586 -0.35686272 -0.35686272 ... -0.372549   -0.35686272
    -0.36470586]
   [-0.38823527 -0.38823527 -0.372549   ... -0.30196077 -0.3098039
    -0.3333333 ]]]]
COMET INFO:   Uploads:
COMET INFO:     environment details      : 1
COMET INFO:     filename                 : 1
COMET INFO:     git metadata             : 1
COMET INFO:     git-patch (uncompressed) : 1 (12.90 KB)
COMET INFO:     images                   : 1
COMET INFO:     installed packages       : 1
COMET INFO:     os packages              : 1
COMET INFO:     source_code              : 1 (6.04 KB)
COMET INFO: 
```



## Development Notes

*   The model implementation in `modeling_gemma.py` (language model) and `modeling_siglip.py` (vision model) aims to be compatible with the downloaded PaliGemma weights. The language model includes features like Rotary Positional Embeddings (RoPE) and Grouped-Query Attention (GQA).
*   The `utils.py` script handles loading the Hugging Face model configuration and `safetensors` weights into the custom PyTorch model. It's currently set to use `strict=True` for `load_state_dict` to catch any discrepancies.
*   `processing_paligemma.py` handles the specific input formatting required, including adding special `<image>` tokens and the `BOS` token.
*   Comet ML integration is present in `inference.py` for tracking experiments. Add your API key, project name, and workspace to environment variables 



## Acknowledgements

*   This project heavily references the architectures of [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) and its components (Gemma and SigLIP).
*   Utilizes the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for tokenizers and pre-trained model access.
*   Experiment tracking by [Comet ML](https://www.comet.com/).