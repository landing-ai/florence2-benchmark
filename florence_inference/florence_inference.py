import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

car_image1 = Image.open("./data/car.jpg")
dog_image1 = Image.open("./data/images_Siberian_Husky.jpg")

# BATCH

images = [dog_image1, car_image1, dog_image1]
prompts = ["<OD>", "<MORE_DETAILED_CAPTION>", "<CAPTION>"]

def run_batch(task_prompts: List[str], images: List[Image.Image], text_inputs: List[str]=None) -> List:
    if text_inputs is None:
        text_inputs = [None] * len(task_prompts)
    
    if len(images) != len(task_prompts):
        raise ValueError("The number of images must match the number of task prompts.")
    
    prompts = []
    
    for task_prompt, text_input in zip(task_prompts, text_inputs):
        if text_input is None:
            prompts.append(task_prompt)
        else:
            prompts.append(task_prompt + text_input)
    
    print(prompts)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=900).to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        num_beams=2
    )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    results = []
    for task_prompt, text, img in zip(task_prompts, generated_texts, images):
        parsed_answer = processor.post_process_generation(text, task=task_prompt, image_size=(img.width, img.height))
        results.append(parsed_answer)
    return results

results = run_batch(prompts, images)

for r in results:
    print(r)
