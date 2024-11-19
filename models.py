import torch
import torch.nn as nn
from transformers import ViTModel, AutoTokenizer, AutoModel, T5EncoderModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='t5-base'):
        super().__init__()
        # T5-base specs: 12-layer, 768-hidden, 12-heads, 272M parameters
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, text):
        text_inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.encoder.parameters()).device)

        text_outputs = self.encoder(**text_inputs).last_hidden_state
        return text_outputs

class SelfAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=12, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_tokens, text_tokens):
        fused_tokens = torch.cat([image_tokens, text_tokens], dim=1)
        attended_tokens, _ = self.attention(fused_tokens, fused_tokens, fused_tokens)
        return self.layer_norm(attended_tokens + fused_tokens)


class ScorePredictor(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 384, kernel_size=2, stride=1, padding=1),
                # nn.LayerNorm(384),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(384, 128, kernel_size=2, stride=1, padding=1),
                # nn.LayerNorm(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(128, 64, kernel_size=2, stride=1, padding=1),
                # nn.LayerNorm(64),
                nn.ReLU()
            )
        ])
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(64, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        
        x = x.transpose(1, 2)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            # Dynamically set LayerNorm dimensions
            if hasattr(conv_layer[1], 'normalized_shape'):
                conv_layer[1].normalized_shape = [x.size(1), x.size(2)]
        
        # Global pooling
        x = self.adaptive_pool(x)  # [batch_size, channels, 1]
        x = x.squeeze(-1)  # [batch_size, channels]
        
        x = self.dense_layers(x)
        return x

class AestheticScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.text_encoder = TextEncoder()
        self.fusion_model = SelfAttentionFusion()
        self.score_predictor = ScorePredictor(input_dim=768)  # ViT hidden dim is 768
        
    def forward(self, images, text):
        image_tokens = self.vit(images).last_hidden_state  # [batch_size, num_patches, hidden_dim]
        # x = vit_output.transpose(1, 2) #TODO Needed?
        # Reshape for conv1d: [batch_size, hidden_dim, num_patches]

        text_tokens = self.text_encoder(text)
        fused_features = self.fusion_model(image_tokens, text_tokens)
        score = self.score_predictor(fused_features)
        return score
