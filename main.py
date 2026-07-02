from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from app.bigram_model import BigramModel
import io
import os
import glob
import torch
import spacy
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from helper_lib.model import get_model

app = FastAPI()

# Load spaCy model with word vectors (medium/large models ship with real
# word embeddings; the small model's vectors are just hashed placeholders)
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"API using device: {device}")

hw2_model = get_model("hw2_cnn").to(device)

# Find best checkpoint
checkpoint_files = sorted(glob.glob("hw2_checkpoints/best/*.pth"))

if len(checkpoint_files) == 0:
    checkpoint_files = sorted(glob.glob("hw2_checkpoints/*.pth"))

if len(checkpoint_files) == 0:
    hw2_model = None
else:
    checkpoint_path = checkpoint_files[-1]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hw2_model.load_state_dict(checkpoint["model_state_dict"])
    hw2_model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load trained GAN generator (final saved model, not per-epoch checkpoints)
hw3_gen_model = get_model("hw3_generator").to(device)
Z_DIM = 100
final_gen_path = "checkpoints/hw3_generator_final.pth"

if not os.path.exists(final_gen_path):
    hw3_gen_model = None
else:
    gen_checkpoint = torch.load(final_gen_path, map_location=device)
    hw3_gen_model.load_state_dict(gen_checkpoint)
    hw3_gen_model.eval()
    print(f"Loaded generator checkpoint: {final_gen_path}")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    text: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class ImageGenerationRequest(BaseModel):
    num_images: int = 16

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    doc = nlp(request.text)
    return {
        "text": request.text,
        "embedding": doc.vector.tolist(),
        "dimensions": doc.vector.shape[0]
    }

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    token1 = nlp(request.word1)
    token2 = nlp(request.word2)
    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": token1.similarity(token2)
    }

@app.post("/generate-image")
def generate_image(request: ImageGenerationRequest):
    if hw3_gen_model is None:
        return {
            "error": "No trained GAN generator checkpoint found. Train the model first."
        }

    noise = torch.randn(request.num_images, Z_DIM).to(device)

    with torch.no_grad():
        fake_images = hw3_gen_model(noise).cpu()

    grid = make_grid(fake_images, normalize=True)
    grid_image = transforms.ToPILImage()(grid)

    buffer = io.BytesIO()
    grid_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if hw2_model is None:
        return {
            "error": "No trained CNN checkpoint found. Train the model first."
        }

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = hw2_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_index = predicted_class.item()
    class_name = CIFAR10_CLASSES[class_index]

    return {
        "predicted_class": class_name,
        "class_index": class_index,
        "confidence": round(confidence.item(), 4)
    }