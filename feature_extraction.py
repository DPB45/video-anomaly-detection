from transformers import ViTFeatureExtractor, ViTModel
import torch

# Load ViT model and extractor
vit_extractor = ViTFeatureExtractor.from_pretrained('facebook/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('facebook/vit-base-patch16-224')

def extract_vit_features(frames):
    inputs = vit_extractor(frames, return_tensors="pt")
    outputs = vit_model(**inputs)
    return outputs.last_hidden_state

if __name__ == "__main__":
    sample_input = torch.rand(1, 3, 224, 224)
    features = extract_vit_features(sample_input)
    print("Extracted Feature Shape:", features.shape)
