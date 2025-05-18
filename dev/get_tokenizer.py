from transformers import AutoConfig

model_id = "google/paligemma-3b-pt-224"
save_path = "/home/poorna/pc_projects/multimodal-vision-language-model/models/paligemma-weights/paligemma-3b-pt-224"

config = AutoConfig.from_pretrained(model_id)
config.save_pretrained(save_path)
