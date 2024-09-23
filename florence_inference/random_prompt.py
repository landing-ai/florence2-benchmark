import base64
from typing import Iterator, Tuple
from random import choice
from pathlib import Path

class RandomLmmPrompt:
    LMM_PROMPTS = {
        '<OCR>': 'pure_text',
        '<OCR_WITH_REGION>': 'ocr',
        '<CAPTION>': 'pure_text',
        '<DETAILED_CAPTION>': 'pure_text',
        '<MORE_DETAILED_CAPTION>': 'pure_text',
        '<OD>': 'description_with_bboxes',
        '<DENSE_REGION_CAPTION>': 'description_with_bboxes',
    }

    # LMM_PROMPTS = {'<OD>': 'description_with_bboxes'}
    def __init__(self, image_folder: str):
        image_folder = Path(image_folder)
        image_files =[ (image_folder / f) for f in image_folder.glob('*.jpg') ]
        self.images = list(map(image_encoding, image_files))
        self.prompts = list(self.LMM_PROMPTS.keys())

    def __call__(self, count:int) -> Iterator[Tuple[str, str]]:
        for i in range(count):
            random_prompt = choice(self.prompts)
            random_image = choice(self.images)
            yield (random_prompt, random_image)


def image_encoding(image_path: str):
    image_encoded_string = None
    with open(image_path, "rb") as image_file:
        image_encoded_string = base64.b64encode(image_file.read()).decode("ascii")
    return image_encoded_string


if __name__ == "__main__":
    random_request = RandomLmmPrompt("./data")
    for prompt, image in random_request(10):
        print(prompt, image[0:10])