import torch
import torch.nn as nn
from transformers import ViTModel, AutoTokenizer, AutoModel, T5EncoderModel, ViTConfig

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

# class AestheticScoreModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vit_encoder = ViTModel.from_pretrained(
#             "google/vit-base-patch16-224-in21k",
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#         )
#         t5_encoder = T5EncoderModel.from_pretrained(
#             "t5-base",
#             hidden_size=768,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=2048,
#         )
#         # self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
#         # self.text_encoder = TextEncoder()
#         self.fusion_model = SelfAttentionFusion()
#         self.score_predictor = ScorePredictor(input_dim=768)  # ViT hidden dim is 768
        
#     def forward(self, images, text):
#         image_tokens = self.vit(images).last_hidden_state  # [batch_size, num_patches, hidden_dim]
#         # x = vit_output.transpose(1, 2) #TODO Needed?
#         # Reshape for conv1d: [batch_size, hidden_dim, num_patches]

#         text_tokens = self.text_encoder(text)
#         import pdb; pdb.set_trace()
#         fused_features = self.fusion_model(image_tokens, text_tokens)
#         score = self.score_predictor(fused_features)
#         return score

class AestheticScoreModel(nn.Module):
    def __init__(self):
        super(AestheticScoreModel, self).__init__()
        vit_config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            patch_size=16,
            image_size=224,
            num_channels=3
        )
        self.vit_encoder = ViTModel(config=vit_config)
        self.vit_encoder.load_state_dict(ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').state_dict())

        # self.vit_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.t5_encoder = T5EncoderModel.from_pretrained('t5-base')
        
        # self.conv_layers = nn.Sequential(
        #     nn.Conv1d(768, 384, kernel_size=2, stride=1),
        #     nn.LayerNorm([384, 246]),
        #     nn.ReLU(),
        #     nn.Conv1d(384, 128, kernel_size=2, stride=1),
        #     nn.LayerNorm([128, 245]),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 64, kernel_size=2, stride=1),
        #     nn.LayerNorm([64, 244]),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=2, stride=1),
        #     nn.LayerNorm([64, 243]),
        #     nn.ReLU()
        # )
        self.conv_layers = nn.Sequential(
            self._conv_block(768, 768, feature_size=247, kernel_size=2, stride=1),
            self._conv_block(768, 384, feature_size=246, kernel_size=2, stride=1),
            self._conv_block(384, 128, feature_size=245, kernel_size=2, stride=1),
            self._conv_block(128, 64, feature_size=244, kernel_size=2, stride=1)
        )
        
        # self.dense_layers = nn.Sequential(
        #     nn.Linear(64, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )

        self.dense_layers = nn.Sequential(
            nn.Linear(64 * 243, 2048),  # 243 = 246 - 3 (due to 4 conv layers with kernel size 2)
            nn.LayerNorm([2048]),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm([1024]),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, feature_size, kernel_size, stride):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.LayerNorm([out_channels, feature_size - (kernel_size - 1)]),
            nn.ReLU()
        )

    def forward(self, image, text):
        image_features = self.vit_encoder(image).last_hidden_state
        # [B, 197, 768] B: Batch size, 197: patches (14*14 patches);  768: hidden dim size
        text_features = self.t5_encoder(input_ids=text).last_hidden_state
        # [B, 50, 768] B: batch size; 512: seq length of tokenized; 768: hidden dim size
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = combined_features.permute(0, 2, 1)  # [8, 768, 247]
        conv_output = self.conv_layers(combined_features)
        conv_output = conv_output.view(conv_output.size(0), -1)
        score = self.dense_layers(conv_output)
        return score

