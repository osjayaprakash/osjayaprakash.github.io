---
layout: post
title: ML Tokenization
permalink: ml-tokenization.html
tags: [machine-learning, llm]
---
* Questions
  * <mark> Write about KV optimizations </mark>
  * <mark> Can we add a new token and learn it effortlessly? </mark>
  * Does tokenization affect multilingual NLP performance (performance and compute)?
  * Why might byte-level tokenization be more robust across languages and domains?
  * How do special tokens (e.g., `[CLS]`, `<s>`, `</s>`, `[PAD]`) influence model training and attention behavior?
  * How does tokenization differ for text, code, and speech data, and why?
  * Why do some tokenizers prefer right padding while others use left padding?
  * What happens if you fine-tune a model with a tokenizer different from the one used during pretraining?
  * How do tokenizers for code-generation models differ from nlp/text generation tokenizers?
  * Why do we need discrete tokenization - continuous, character-based, or byte-embedding approaches?
  * Is it possible to have self-tokenizers or adaptive tokenizers in model architecture?

* breaks down text (or other data) into smaller units called tokens (ints) before passing them into a ml model
  * machines don't understand the text, they can work with numbers
  * Embedding Layer converts the token ids into dense vectors (embeddings).
  * Embeddings capture semantic meaning
* tokens are not necessarily words (it used to be)
  * subwords (e.g. “play” + “ing”)
  * characters (e.g. “p” + “l” + “a” + “y”)
  * even bytes (for multilingual handling and non-texts handling)
* LLMs (like GPT, Claude, Gemini) typically use BPE or SentencePiece. They are flexible - no OOV
* most common tokenizers
  * BPE - byte pair encoding
  * Wordpiece
  * Sentencepiece
* number of unique token ids (vocabulary size) is between 30,000 to 100,000
* if the vocab is too small, then the words are broken down and split into longer sequences.
  * the model must learn and longers inputs for text!
  * more tokens is more compute
  * worst semantic understanding ~ assume if a word is broken into individual chars - which may not itself carry useful information.
* if the vocab is too large, then vocabulary includes lot of rarely used tokens and duplicates/similar tokens.
  * having token for every number mentioned in the webpage, dates, phone numbers, address etc!!
  * similar words might have different token ids - tokenize, tokenization, tokenizing, tokenized
  * poor learning, generalization
* special tokens exist for -
  * `[PAD]` – padding
  * `[UNK]` – unknown / out-of-vocab
  * `[CLS]` – start-of-sequence (classification token)
  * `[SEP]` – separator between sentences
  * `[MASK]` – masked token for MLM
  * `<bos>` / `<eos>` – beginning/end of sequence (GPT-style models)
  * System/user/assistant role tokens in chat models
  * Coding agents some special tokens for reserved words in languages (including tabs for pythons)
* padding or truncation
  * right padding is most common and default in hf. used especially in classification, seq2seq, BERT/T5 ~
  * left padding is useful for batched autoregressive generation with decoder-only models (GPT-like, code models, Qwen, LLaMA, etc.). improves KV-Optimizations.
    * most recent tokens (at the end) are most relevant. In autoregressive generation, the model only “looks left.”
The latest tokens are the most informative for predicting the next word.

* Papers
  * [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
  * [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
  * [Fast WordPiece Tokenization](https://aclanthology.org/2021.emnlp-main.160)
  * [Tokenization Is More Than Compression](https://aclanthology.org/2024.emnlp-main.40)
  * [Wikipedia about BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding)
  * [Byte Pair Encoding is Suboptimal for Language Model Pretraining]
  * [Kaggle Intro](https://www.kaggle.com/code/william2020/how-openai-s-byte-pair-encoding-bpe-works)

* BPE
  * todo

* EXAMPLE -

```py
from transformers import AutoTokenizer

#   "codellama/CodeLlama-7b-hf"
#   "bigcode/starcoder2-3b"
#   "deepseek-ai/deepseek-coder-6.7b-base"
model_name = "bert-base-uncased"
# model_name = "deepseek-ai/deepseek-coder-6.7b-base"
tok = AutoTokenizer.from_pretrained(model_name)

print(tok)

print("All special tokens:", tok.all_special_tokens)
print("All special IDs:", tok.all_special_ids)
print("Special token map:", tok.special_tokens_map)
print("Base vocab size:", tok.vocab_size)
print("Total tokens (including added):", len(tok))

text = "Tokenization is Cool! 😎"
# text = """
#     import numpy as np;
#     import pandas as pd;
#     import matplotlib.pyplot as plt;
#     import seaborn as sns;

#     a = np.zeros((2,10))
#     b = 1.001
# """

tokens = tok.tokenize(text)
print("Token texts:", tokens)

token_ids = tok.encode(text, add_special_tokens=True)
print("Token ids:", token_ids)
tokens = tok.convert_ids_to_tokens(token_ids)
print("Token texts from ids:", tokens)

```

* bert-base-uncased

```
BertTokenizerFast(name_or_path='bert-base-uncased', vocab_size=30522, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)
All special tokens: ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
All special IDs: [100, 102, 0, 101, 103]
Special token map: {'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
Base vocab size: 30522
Total tokens (including added): 30522
Token texts: ['token', '##ization', 'is', 'cool', '!', '[UNK]']
Token ids: [101, 19204, 3989, 2003, 4658, 999, 100, 102]
Token texts from ids: ['[CLS]', 'token', '##ization', 'is', 'cool', '!', '[UNK]', '[SEP]']
```

* deepseek-ai/deepseek-coder-6.7b-base

```
LlamaTokenizerFast(name_or_path='deepseek-ai/deepseek-coder-6.7b-base', vocab_size=32000, model_max_length=16384, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<｜begin▁of▁sentence｜>', 'eos_token': '<｜end▁of▁sentence｜>', 'pad_token': '<｜end▁of▁sentence｜>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	32000: AddedToken("õ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32001: AddedToken("÷", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32002: AddedToken("Á", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32003: AddedToken("ý", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32004: AddedToken("À", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32005: AddedToken("ÿ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32006: AddedToken("ø", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32007: AddedToken("ú", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32008: AddedToken("þ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32009: AddedToken("ü", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32010: AddedToken("ù", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32011: AddedToken("ö", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32012: AddedToken("û", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32013: AddedToken("<｜begin▁of▁sentence｜>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	32014: AddedToken("<｜end▁of▁sentence｜>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	32015: AddedToken("<｜fim▁hole｜>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32016: AddedToken("<｜fim▁begin｜>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32017: AddedToken("<｜fim▁end｜>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32018: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32019: AddedToken("<|User|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32020: AddedToken("<|Assistant|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	32021: AddedToken("<|EOT|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
}
)
All special tokens: ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>']
All special IDs: [32013, 32014]
Special token map: {'bos_token': '<｜begin▁of▁sentence｜>', 'eos_token': '<｜end▁of▁sentence｜>', 'pad_token': '<｜end▁of▁sentence｜>'}
Base vocab size: 32000
Total tokens (including added): 32022
Token texts: ['Ċ', 'ĠĠĠ', 'Ġimport', 'Ġnum', 'py', 'Ġas', 'Ġnp', ';', 'Ġ', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġpand', 'as', 'Ġas', 'Ġp', 'd', ';', 'Ġ', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġmat', 'plot', 'lib', '.', 'py', 'plot', 'Ġas', 'Ġpl', 't', ';', 'Ġ', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġse', 'ab', 'orn', 'Ġas', 'Ġs', 'ns', ';', 'Ċ', 'Ċ', 'ĠĠĠ', 'Ġa', 'Ġ=', 'Ġnp', '.', 'zer', 'os', '((', '2', ',', '1', '0', '))', 'Ċ', 'ĠĠĠ', 'Ġb', 'Ġ=Ġ', '1', '.', '0', '0', '1', 'Ċ']
Token ids: [32013, 185, 315, 1659, 1181, 4016, 372, 21807, 26, 207, 185, 315, 1659, 21866, 281, 372, 265, 67, 26, 207, 185, 315, 1659, 1575, 13371, 2875, 13, 4016, 13371, 372, 568, 83, 26, 207, 185, 315, 1659, 386, 356, 1745, 372, 252, 3585, 26, 185, 185, 315, 245, 405, 21807, 13, 9888, 378, 5930, 17, 11, 16, 15, 1435, 185, 315, 270, 1412, 16, 13, 15, 15, 16, 185]
Token texts from ids: ['<｜begin▁of▁sentence｜>', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġnum', 'py', 'Ġas', 'Ġnp', ';', 'Ġ', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġpand', 'as', 'Ġas', 'Ġp', 'd', ';', 'Ġ', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġmat', 'plot', 'lib', '.', 'py', 'plot', 'Ġas', 'Ġpl', 't', ';', 'Ġ', 'Ċ', 'ĠĠĠ', 'Ġimport', 'Ġse', 'ab', 'orn', 'Ġas', 'Ġs', 'ns', ';', 'Ċ', 'Ċ', 'ĠĠĠ', 'Ġa', 'Ġ=', 'Ġnp', '.', 'zer', 'os', '((', '2', ',', '1', '0', '))', 'Ċ', 'ĠĠĠ', 'Ġb', 'Ġ=Ġ', '1', '.', '0', '0', '1', 'Ċ']
```

* Qwen/Qwen1.5-1.8B-Chat

```
Qwen2TokenizerFast(name_or_path='Qwen/Qwen1.5-1.8B-Chat', vocab_size=151643, model_max_length=32768, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)
All special tokens: ['<|im_end|>', '<|endoftext|>', '<|im_start|>']
All special IDs: [151645, 151643, 151644]
Special token map: {'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}
Base vocab size: 151643
Total tokens (including added): 151646
Token texts: ['Token', 'ization', 'Ġis', 'ĠCool', '!', 'ĠðŁĺ', 'İ']
Token ids: [3323, 2022, 374, 23931, 0, 26525, 236]
Token texts from ids: ['Token', 'ization', 'Ġis', 'ĠCool', '!', 'ĠðŁĺ', 'İ']
```




