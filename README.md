# SARCHv1: AI Text Generation Model

## Overview
SARCHv1 is a text generation AI built using the `TFGPT2LMHeadModel` from the Hugging Face Transformers library. This project focuses on creating a custom AI text generator, leveraging the GPT-2 model with TensorFlow for flexible task adaptation.

## Features
- **Custom task adaptation**: Adapts GPT-2 outputs to fit specialized use cases.
- **Configurable tokenization and model loading**: Allows users to input text and generate AI responses.
- **Error handling**: Includes checks for tokenization size mismatches and dimension alignment.

## Installation

Ensure you have Python 3.12 installed. Then, install the necessary dependencies:

```bash
pip install transformers tensorflow
```

Optionally, if you are using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## File Structure

- `sarchv1.py`: The main script containing the AI model logic.

## Usage

### 1. Prepare the Model
Ensure your `config.json` is correctly set up with the GPT-2 model name and tokenizer.

Example `config.json`:

```json
{
    "model_name": "gpt2",
    "tokenizer_name": "gpt2",
    "max_length": 100
}
```

### 2. Run the Model
To generate text, run the following command:

```bash
python sarchv1.py --input "Your custom input text here."
```

#### Command-line Arguments

- `--input`: (Required) The text input for the AI to process.
- `--max_length`: (Optional) The maximum length of the generated output (default: 100).
- `--temperature`: (Optional) Controls randomness — lower values make output more deterministic (default: 1.0).

### 3. Model Initialization

The model is initialized in `sarchv1.py` with:

```python
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

model = TFGPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

Ensure the correct model path is provided if you are using a custom fine-tuned GPT-2.

### 4. Text Generation Workflow

The process follows these steps:

1. **Tokenization**:
   ```python
   tokenized_input = tokenizer(input_text, return_tensors='tf')
   ```
2. **Model Call**:
   ```python
   output = model.call(inputs=tokenized_input)
   ```
3. **Decoding**:
   ```python
   generated_text = tokenizer.decode(output.logits[0], skip_special_tokens=True)
   print(generated_text)
   ```

## Troubleshooting

### Dimension Mismatch Error

If you encounter an error like:

```
ValueError: Dimensions must be equal, but are 50257 and 768
```

It likely means there's a shape misalignment in `adapt_to_task`. Ensure you're correctly processing the output logits and reshaping tensors.

Check this section of `sarchv1.py`:

```python
second_pass = model(inputs=tokenized_input).logits
fast_output = some_processing_function(second_pass)
output = second_pass + fast_output
```

Ensure both `second_pass` and `fast_output` have compatible shapes.

## Contributing

Feel free to fork the repository and submit pull requests. Make sure to add detailed commit messages and test your changes thoroughly.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Contact

For questions or support, please reach out to the project maintainer.

---

Happy coding!

