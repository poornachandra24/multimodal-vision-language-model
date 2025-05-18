
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

3.  **Download Model Weights and Tokenizer:**
    The model architecture is designed to be compatible with pre-trained PaliGemma weights. You need to download these from Hugging Face Hub.

    *   **Option A (Recommended - Manual Download Script):**
        Create a Python script (e.g., `download_model.py` in the project root) with the following content:
        ```python
        from transformers import AutoTokenizer, AutoModelForPreTraining # Using AutoModelForPreTraining for generic download
        import os

        model_id = "google/paligemma-3b-pt-224"
        save_path = os.path.join("models", "paligemma-weights", "paligemma-3b-pt-224")

        print(f"Ensuring save directory exists: {save_path}")
        os.makedirs(save_path, exist_ok=True)

        print(f"Downloading tokenizer for {model_id} to {save_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer downloaded and saved.")

        print(f"Downloading model weights for {model_id} to {save_path}...")
        # Note: This might download the full model class if available,
        # or you can use snapshot_download for more control if needed for very large models.
        # For PaliGemma, AutoModelForPreTraining should work for downloading all files.
        model = AutoModelForPreTraining.from_pretrained(model_id)
        model.save_pretrained(save_path) # This saves config.json and weight files (.safetensors)
        print("Model weights and config downloaded and saved.")
        print(f"All files should now be in: {save_path}")
        ```
        Then run this script:
        ```bash
        python download_model.py
        ```

    *   **Option B (Hugging Face CLI - Requires login for gated Models):**
        Make sure you are logged into Hugging Face CLI:
        ```bash
        huggingface-cli login
        ```
        Then, you might need to use `snapshot_download` or manually ensure all files are present. The script in Option A is generally more reliable for getting all necessary components. You can run `get_tokenizer.py`

    After running the download script, your `models/paligemma-weights/paligemma-3b-pt-224/` directory should contain:
    *   `config.json`
    *   `tokenizer.json` (or `tokenizer.model`)
    *   `special_tokens_map.json`
    *   `tokenizer_config.json`
    *   Multiple `.safetensors` files (the model weights)

4.  **Prepare a Test Image:**
    Place a test image (e.g., `image.png`) in the `dev/test_image/` directory. The `launch_inference.sh` script is configured to use `dev/test_image/image.png` by default.

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

## Development Notes

*   The model implementation in `modeling_gemma.py` (language model) and `modeling_siglip.py` (vision model) aims to be compatible with the downloaded PaliGemma weights. Debugging focuses on ensuring correct layer naming and architectural correspondence.
*   The `utils.py` script handles loading the Hugging Face model configuration and `safetensors` weights into the custom PyTorch model. It's currently set to use `strict=True` for `load_state_dict` to catch any discrepancies.
*   `processing_paligemma.py` handles the specific input formatting required, including adding special `<image>` tokens and the `BOS` token.



## Acknowledgements

*   This project heavily references the architectures of [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) and its components (Gemma and SigLIP).
*   Utilizes the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for tokenizers and pre-trained model access.