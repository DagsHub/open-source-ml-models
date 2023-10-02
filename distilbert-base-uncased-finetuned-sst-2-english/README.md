![Cover](https://user-images.githubusercontent.com/66431403/267597217-2285216a-209c-466b-a0da-ab610356c2af.png)

# distilbert-base-uncased-finetuned-sst-2-english

## DagsHub Repository: https://dagshub.com/Rutam21/distilbert-base-uncased-finetuned-sst-2-english

## Source: [HuggingFace distilbert-base-uncased-finetuned-sst-2-english Model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

# Description

This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2. This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).

# Getting Started with the Model

**Example of single-label classification:**

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

```

# Training

## Training Data

The authors use the following Stanford Sentiment Treebank([sst2](https://huggingface.co/datasets/sst2)) corpora for the model.

## Training Procedure

### Fine-tuning hyper-parameters


- learning_rate = 1e-5
- batch_size = 32
- warmup = 600
- max_seq_length = 128
- num_train_epochs = 3.0

# Usage

## Direct Use

This model can be used for  topic classification. You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you.

## Misuse and Out-of-scope Use
The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.


# Risks, Limitations and Biases

Based on a few experimentations, we observed that this model could produce biased predictions that target underrepresented populations.

For instance, for sentences like `This film was filmed in COUNTRY`, this binary classification model will give radically different probabilities for the positive label depending on the country (0.89 if the country is France, but 0.08 if the country is Afghanistan) when nothing in the input indicates such a strong semantic shift. In this [colab](https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb), [Aurélien Géron](https://twitter.com/aureliengeron) made an interesting map plotting these probabilities for each country.

<img src="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/map.jpeg" alt="Map of positive probabilities per country." width="500"/>

We strongly advise users to thoroughly probe these aspects on their use-cases in order to evaluate the risks of this model. We recommend looking at the following bias evaluation datasets as a place to start: [WinoBias](https://huggingface.co/datasets/wino_bias), [WinoGender](https://huggingface.co/datasets/super_glue), [Stereoset](https://huggingface.co/datasets/stereoset).

# License

This model is available on HuggingFace under the Apache-2.0 License.

# Citation

```citation

  author    = {Victor SANH and
               Lysandre DEBUT and
               Julien CHAUMOND and
               Thomas WOLF},
  title     = {DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  year      = {2019},
  url       = {https://arxiv.org/abs/1910.01108},
  archivePrefix = {arXiv},
  eprint    = {1910.01108},
  timestamp = {Wed, 2 Oct 2019 17:56:28 UTC}
```
