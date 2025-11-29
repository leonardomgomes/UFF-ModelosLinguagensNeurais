# pipeline_com_llm.py
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cpu")
DTYPE = torch.float32

LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"

print(">> Carregando modelos (COM LLM)...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=DTYPE,
    device_map="cpu"
)

pipe = StableDiffusionPipeline.from_pretrained(
    IMAGE_MODEL,
    torch_dtype=DTYPE
).to("cpu")


def melhorar_prompt(texto: str) -> str:
    prompt = f"Melhore o seguinte prompt para geração de imagem, sem inventar nada além do necessário: {texto}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def gerar_imagem_com_llm(texto: str, emocoes: list, num_img: int = 1):
    entrada = f"{texto}. A atmosfera do ambiente na cena e o tom emocional são de {', '.join(emocoes)}."
    prompt_otimizado = melhorar_prompt(entrada)

    images = []
    for _ in range(num_img):
        img = pipe(prompt_otimizado).images[0]
        images.append(img)

    return prompt_otimizado, images
