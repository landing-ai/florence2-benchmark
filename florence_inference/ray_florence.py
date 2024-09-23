import io
from ray import serve
import torch
from base64 import decodebytes
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from starlette.requests import Request
from typing import Dict
from typing import List

@serve.deployment(name="florence2")
class Florence2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large",
                                                          torch_dtype=self.torch_dtype, 
                                                          trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large",
                                                       trust_remote_code=True)

    def _parse_image(self, encoded_image) -> Image.Image:
        return Image.open(io.BytesIO(decodebytes(bytes(encoded_image, "ascii"))))

    def generate(self, task_prompts: List[str], text_inputs: List[str], images: List[Image.Image]) -> List[str]:
        # Run inference
        prompts = []
        for task_prompt, text_input in zip(task_prompts, text_inputs):
            if text_input is None:
                prompts.append(task_prompt)
            else:
                prompts.append(task_prompt + text_input)

        inputs = self.processor(text=prompts, images=images, return_tensors="pt", 
                                padding=True, truncation=True, max_length=900).to(self.device, dtype=self.torch_dtype)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        results = []
        for task_prompt, text, img in zip(task_prompts, generated_texts, images):
            parsed_answer = self.processor.post_process_generation(text, task=task_prompt, image_size=(img.width, img.height))
            results.append(parsed_answer)
        return results
    

    def reconfigure(self, user_config: Dict):
        self.__call__.set_max_batch_size(user_config["max_batch_size"])
        self.__call__.set_batch_wait_timeout_s(user_config["batch_wait_timeout_s"])

    @serve.batch(max_batch_size=5, batch_wait_timeout_s=0.1)
    async def __call__(self, http_requests: List[Request]) -> List[str]:
        responses = []
        tasks = []
        text_inputs = []
        images = []
        for index, http_request in enumerate(http_requests):
            request_json = await http_request.json()
            task_prompt = request_json.get("task_prompt")
            text_input = request_json.get("text_input")
            encoded_image = request_json.get("image")
            if not task_prompt or not encoded_image:
                responses.append("task_prompt and image field is mandatory in request JSON!")
                continue
            try: 
                image = self._parse_image(encoded_image)
            except Exception as e:
                responses.append("image base64 encoding is invalid")
                continue
            # request = (index, task_prompt, text_input, encoded_image)
            # add request into request groups based on task_prompt
            tasks.append(task_prompt)
            text_inputs.append(text_input)
            images.append(image)

        # reorder group reponse to follow the same order as the 
        responses = self.generate(tasks, text_inputs, images)                           
        # make sure the request and reponse are in the same order http_requests
        return responses


# create florence2 app
lmm_app = Florence2.bind()
