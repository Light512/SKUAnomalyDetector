
import torch
from transformers import AutoImageProcessor, AutoModel

# Configuration
IMAGE_ROOT = "img" 
OUTPUT_CSV = "clustered_output.csv"
MODEL_TYPE = 'facebook/dinov2-base'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
image_processor = AutoImageProcessor.from_pretrained(MODEL_TYPE)
model = AutoModel.from_pretrained(MODEL_TYPE).to(DEVICE)
model.eval()
