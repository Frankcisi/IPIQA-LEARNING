import torch
import clip
from PIL import Image
decice="cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=decice)
model.eval()
prompts = [
    "a high quality image",
    "a low quality image",
    "a sharp image",
    "a blurry image",
    "a noisy image"
]
# 3. text -> token -> text feature
text_tokens = clip.tokenize(prompts).to(decice)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)
print("text features shape:", text_features.shape)
image_paths =[ "test.png","test01.png"]
for image_path in image_paths:
    image=preprocess(Image.open(image_path)).unsqueeze(0).to(decice)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    print("image features shape:", image_features.shape)
    similarity = image_features @ text_features.T
    print(f"\nImage: {image_path}")
    for prompt, score in zip(prompts, similarity[0]):
        print(f"{prompt}: {score.item():.4f}")