![Cover](https://user-images.githubusercontent.com/66431403/267597217-2285216a-209c-466b-a0da-ab610356c2af.png)

# vilt-b32-finetuned-vqa

## DagsHub Repository: https://dagshub.com/Rutam21/vilt-b32-finetuned-vqa

## Source: [HuggingFace vilt-b32-finetuned-vqa Model](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)

# Description

Vision-and-Language Transformer (ViLT) model fine-tuned on [VQAv2](https://visualqa.org/). It was introduced in the paper [ViLT: Vision-and-Language Transformer
Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) by Kim et al. and first released in [this repository](https://github.com/dandelin/ViLT). 

![ViLT](https://raw.githubusercontent.com/dandelin/ViLT/master/assets/vilt.png)

# Usage

Here is how to use this model in PyTorch:

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```

# Pretrained Weights
You can find five pretrained weights below.

1. ViLT-B/32 Pretrained with MLM+ITM for 200k steps on GCC+SBU+COCO+VG (ViLT-B/32 200k) [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)
2. ViLT-B/32 200k finetuned on VQAv2 [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_vqa.ckpt)
3. ViLT-B/32 200k finetuned on NLVR2 [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_nlvr2.ckpt)
4. ViLT-B/32 200k finetuned on COCO IR/TR [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_coco.ckpt)
5. ViLT-B/32 200k finetuned on F30K IR/TR [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_f30k.ckpt)

# License

This model is available on HuggingFace under the Apache-2.0 License.

# Citation

```citation
@misc{kim2021vilt,
      title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}, 
      author={Wonjae Kim and Bokyung Son and Ildoo Kim},
      year={2021},
      eprint={2102.03334},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
