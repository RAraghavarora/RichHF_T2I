import torchvision.transforms as transforms
from PIL import Image
import io


def bytes_to_tensor(image_bytes):
    """Convert image bytes to normalized tensor suitable for ViT"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT typically expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    return transform(image)

def tokenize_captions(captions, tokenizer, max_length=77):
    return tokenizer(
        captions,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * math.sqrt(warmup_steps / step)