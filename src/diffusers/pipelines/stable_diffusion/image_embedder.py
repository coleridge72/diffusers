import torch
import torch.nn as nn

from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


class StableDiffusionImageEmbedder(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

    @torch.no_grad()
    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        return self.visual_projection(pooled_output)

    @torch.no_grad()
    def forward_onnx(self, clip_input: torch.FloatTensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        return self.visual_projection(pooled_output)