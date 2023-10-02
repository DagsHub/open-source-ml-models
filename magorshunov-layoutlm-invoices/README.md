![Cover](https://user-images.githubusercontent.com/66431403/267597217-2285216a-209c-466b-a0da-ab610356c2af.png)

# layoutlm-invoices

## DagsHub Repository: https://dagshub.com/Rutam21/layoutlm-invoices

## Source: [HuggingFace layoutlm-invoices Model](https://huggingface.co/magorshunov/layoutlm-invoices)

# Description

This is a fine-tuned version of the multi-modal LayoutLM model for the task of question answering on invoices and other documents. It has been fine-tuned on a proprietary dataset of invoices as well as both SQuAD2.0 and DocVQA for general comprehension.

# Fine Tune Results

## Non-consecutive tokens

Unlike other QA models, which can only extract consecutive tokens (because they predict the start and end of a sequence), this model can predict longer-range, non-consecutive sequences with an additional classifier head. For example, QA models often encounter this failure mode:

### Before

![Before Results](https://dagshub.com/Rutam21/layoutlm-invoices/raw/main/before.png)

### After

However this model is able to predict non-consecutive tokens and therefore the address correctly.

![After Results](https://dagshub.com/Rutam21/layoutlm-invoices/raw/main/after.png)

# License

This model is available on HuggingFace under the CC by NC-SA 4.0 License.

# Citation

```citation
This model was created by the team at Impira.
```
