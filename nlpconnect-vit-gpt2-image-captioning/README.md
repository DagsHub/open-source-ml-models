![Cover](https://user-images.githubusercontent.com/66431403/267597217-2285216a-209c-466b-a0da-ab610356c2af.png)

## DagsHub Repository: https://dagshub.com/Rutam21/vit-gpt2-image-captioning
## Source: [HuggingFace vit-gpt2-image-captioning Model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

# nlpconnect/vit-gpt2-image-captioning

The Vision Encoder Decoder Model can be used to initialize an image-to-text model with any pre-trained Transformer-based vision model as the encoder (e.g. ViT, BEiT, DeiT, Swin) and any pre-trained language model as the decoder (e.g. RoBERTa, GPT2, BERT, DistilBERT).

Image captioning is an example, in which the encoder model is used to encode the image, after which an autoregressive language model i.e. the decoder model generates the caption.


# Illustrated Image Captioning using transformers

![Transformers](https://ankur3107.github.io/assets/images/vision-encoder-decoder.png)



# Sample Code

```python

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


predict_step(['doctor.e16ba4e4.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']

```

# Sample Code using Transformers

```python

from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]


```

# License
This model is licensed under Apache-2.0 on HuggingFace.

# References
- https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder
- https://huggingface.co/docs/transformers/model_doc/encoder-decoder
- https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2
- https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
# Citation
```citation
@article{kumar2022imagecaptioning,
title   = "The Illustrated Image Captioning using transformers",
author  = "Kumar, Ankur",
journal = "ankur3107.github.io",
year    = "2022",
url     = "https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/"
}
```
