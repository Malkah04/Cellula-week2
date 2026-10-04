import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:

    def __init__(self,model_name="Salesforce/blip-image-captioning-base"):
        
        self.device = torch.device("cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate_caption(self, image):
        if image is None: raise ValueError("Image is required.")

        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        image = image.convert("RGB")
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=3
            )
        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )
        return caption.strip()