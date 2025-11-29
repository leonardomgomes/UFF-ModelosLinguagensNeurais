# pipeline_sem_llm.py
import torch
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
EMBED_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"

print(">> Carregando modelos (SEM LLM)...")

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

pipe = StableDiffusionPipeline.from_pretrained(
    IMAGE_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)


def gerar_imagem_sem_llm(texto: str, emocoes: list, num_img: int = 1):
    entrada_concat = f"{texto}. A atmosfera do ambiente na cena e o tom emocional são de {', '.join(emocoes)}."

    images = []
    for _ in range(num_img):
        result = pipe(
            entrada_concat,
            num_inference_steps=50,
            guidance_scale=7.5
        )
        images.append(result.images[0])

    return images
