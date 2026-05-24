# 2026-05-24 Text Encoder Demo

## Today’s Goal

Today I started exploring the CLIP Text Encoder part in IPIQA and implemented a minimal image-text similarity demo. The main goal was to understand how CLIP encodes text prompts, understand feature vectors and similarity computation, and verify whether CLIP can capture quality-related semantics from prompts.

## Understanding the CLIP Text Encoder Pipeline

Today I learned the basic CLIP image-text pipeline:

```text
prompt
→ tokenize
→ text encoder
→ text feature vector
```

and:

```text
image
→ image encoder
→ image feature vector
```

Finally:

```text
image feature
↔ text feature
```

through cosine similarity.

## Quality-related Prompts

I manually designed several quality-related prompts:

```python
prompts = [
    "a high quality image",
    "a low quality image",
    "a sharp image",
    "a blurry image",
    "a noisy image"
]
```

These prompts were used to test whether CLIP can understand quality semantics.

## Tokenization and Text Feature Extraction

I used:

```python
clip.tokenize(prompts)
```

to convert prompts into tokens.

Then I used:

```python
model.encode_text()
```

to obtain text semantic features.

Output:

```python
text features shape: [5, 512]
```

Meaning:

- 5 prompts
- each prompt represented by a 512-dimensional semantic vector

## Image Feature Extraction

I used:

```python
model.encode_image()
```

to extract image semantic features.

Output:

```python
image features shape: [1, 512]
```

Meaning:

- 1 image
- represented by a 512-dimensional image feature vector

## Feature Normalization

I used:

```python
feature /= feature.norm(dim=-1, keepdim=True)
```

to normalize feature vectors.

Today I learned:

- `.norm()` computes vector length
- normalization converts vectors into unit vectors
- CLIP focuses more on feature direction (semantic similarity) instead of feature magnitude

## Image-Text Similarity

I computed similarity using:

```python
similarity = image_features @ text_features.T
```

Today I learned:

- `@` means matrix multiplication
- after normalization, dot product becomes cosine similarity
- cosine similarity measures semantic closeness in feature space

## Experimental Results

### Distorted Image 1

| Prompt | Similarity |
|---|---:|
| a high quality image | 0.2089 |
| a low quality image | 0.2280 |
| a sharp image | 0.2357 |
| a blurry image | 0.2606 |
| a noisy image | 0.2527 |

Top prompts:

```text
a blurry image
a noisy image
```

### Distorted Image 2

| Prompt | Similarity |
|---|---:|
| a high quality image | 0.2021 |
| a low quality image | 0.2012 |
| a sharp image | 0.2345 |
| a blurry image | 0.2156 |
| a noisy image | 0.2240 |

Top prompt:

```text
a sharp image
```

## Observations

Compared with Distorted Image 2, Distorted Image 1 obtained higher similarity scores on:

- `"a blurry image"`
- `"a noisy image"`

while Distorted Image 2 obtained the highest score on:

- `"a sharp image"`

This indicates that CLIP can capture basic quality-related semantics through image-text similarity.

## What I Learned Today

Today I gained a deeper understanding of:

- CLIP text encoder
- tokenization
- feature vectors
- cosine similarity
- feature normalization
- image-text alignment
- representation learning

I also realized that:

```text
modern IQA is not only regression,
but also semantic alignment.
```

## Next Step

Possible next steps:

- add more quality prompts
- try prompt engineering
- test more distortion types
- start implementing the full IPIQA framework
- explore how text semantics can guide IQA prediction