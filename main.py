# main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import time
from datetime import datetime
import threading
import traceback

# Tente importar os módulos de pipeline e emotion (se você os criou conforme sugestões)
try:
    from emotion_model import prever_emocoes, EMOCOES  # caso tenha criado emotion_model.py
except Exception:
    prever_emocoes = None
    EMOCOES = [
        'admiração', 'diversão', 'raiva', 'irritação', 'aprovação', 'carinho',
        'confusão', 'curiosidade', 'desejo', 'decepção', 'desaprovação', 'nojo',
        'constrangimento', 'empolgação', 'medo', 'gratidão', 'pesar', 'alegria',
        'amor', 'nervosismo', 'otimismo', 'orgulho', 'percepção', 'alívio',
        'remorso', 'tristeza', 'surpresa', 'neutro'
    ]

try:
    # módulos que você forneceu: assumem interfaces geráveis
    import pipeline_sem_llm as pipe_sem
except Exception:
    pipe_sem = None

try:
    import pipeline_com_llm as pipe_com
except Exception:
    pipe_com = None

# ===========================================================
# CONFIGURAÇÕES
# ===========================================================
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

os.makedirs(os.path.join(STATIC_DIR, "outputs_com_llm"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "outputs_sem_llm"), exist_ok=True)

# ===========================================================
# SESSÃO SIMPLES (memória em processo)
# ===========================================================
SESSAO = {
    "texto": "",
    "emocoes_previstas": {},
    "texto_completo": "",
    # geração
    "in_progress": False,
    "progress": 0,
    "imagens_sem_llm": [],
    "imagens_com_llm": [],
    "tempos_sem_llm": [],
    "tempos_com_llm": [],
    "prompt_otimizado": "",
    "erro_geracao": None,
}

progress_lock = threading.Lock()


# ===========================================================
# HOME
# ===========================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===========================================================
# ETAPA 1 → PROCESSAR EMOÇÕES
# ===========================================================
@app.post("/processar", response_class=HTMLResponse)
async def processar(
    request: Request,
    user_text: str = Form(...),
    modo_geracao_hidden: str = Form("ambos"),
    modo_acoes_hidden: str = Form("manual")
):
    if not user_text.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "erro": "Por favor, digite um texto antes de processar.",
            },
        )

    # ----- inferência real se disponível -----
    if prever_emocoes is not None:
        try:
            emocoes_previstas = prever_emocoes(user_text)
        except Exception:
            traceback.print_exc()
            # fallback: zeros
            emocoes_previstas = {emo: 0.0 for emo in EMOCOES}
    else:
        # fallback temporário (simulação)
        modelo_saida = {
            "alegria": 0.92,
            "otimismo": 0.74,
            "raiva": 0.12,
            "tristeza": 0.05
        }
        emocoes_previstas = {emo: modelo_saida.get(emo, 0.0) for emo in EMOCOES}

    emocoes_relevantes = [e for e, p in emocoes_previstas.items() if p > 0.5]
    texto_completo = f"{user_text}. Emoções detectadas: {', '.join(emocoes_relevantes)}" if emocoes_relevantes else f"{user_text}. Emoções detectadas: nenhuma relevante"

    # SALVA NA SESSÃO
    SESSAO["texto"] = user_text
    SESSAO["emocoes_previstas"] = emocoes_previstas
    SESSAO["texto_completo"] = texto_completo

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_text": user_text,
            "texto_completo": texto_completo,
            "emocoes": emocoes_previstas,
            "modo_geracao": modo_geracao_hidden,
            "modo_acoes": modo_acoes_hidden,
            "apenas_emocoes": True,
            "mostrar_form_gerar_imagens": True
        }
    )


# ===========================================================
# BACKGROUND TASK: geração de imagens
# ===========================================================
def _gera_imagens_background(modo_geracao: str, num_imgs: int = 3):
    """
    Função executada em background thread. Atualiza SESSAO com progresso e resultados.
    """
    try:
        with progress_lock:
            SESSAO["in_progress"] = True
            SESSAO["progress"] = 0
            SESSAO["imagens_sem_llm"] = []
            SESSAO["imagens_com_llm"] = []
            SESSAO["tempos_sem_llm"] = []
            SESSAO["tempos_com_llm"] = []
            SESSAO["prompt_otimizado"] = ""
            SESSAO["erro_geracao"] = None

        user_text = SESSAO["texto"]
        emocoes_previstas = SESSAO["emocoes_previstas"]
        # seleciona emoções relevantes (p>0.5) ou pega top-2
        emocoes_relevantes = [e for e, p in emocoes_previstas.items() if p > 0.5]
        if not emocoes_relevantes:
            # pega top-2
            sorted_em = sorted(emocoes_previstas.items(), key=lambda x: x[1], reverse=True)
            emocoes_relevantes = [e for e, p in sorted_em[:2] if p > 0]

        total_steps = 0
        steps_done = 0
        if modo_geracao in ("sem_llm", "ambos"):
            total_steps += num_imgs
        if modo_geracao in ("com_llm", "ambos"):
            total_steps += num_imgs + 1  # +1 para otimização do prompt pelo LLM

        # ---------------- SEM LLM ----------------
        if modo_geracao in ("sem_llm", "ambos"):
            for i in range(num_imgs):
                start = time.time()
                try:
                    if pipe_sem is not None and hasattr(pipe_sem, "gerar_imagem_sem_llm"):
                        # espera que gerar_imagem_sem_llm retorne PIL.Image objects list or single image
                        imgs = pipe_sem.gerar_imagem_sem_llm(user_text, emocoes_relevantes, num_img=1)
                        # suporte a lista ou único
                        img = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
                    else:
                        # fallback: criar arquivo PNG simples
                        from PIL import Image, ImageDraw, ImageFont
                        img = Image.new("RGB", (512, 512), color=(200, 200, 200))
                        d = ImageDraw.Draw(img)
                        d.text((20, 20), f"SEM LLM\n{i}", fill=(0, 0, 0))
                except Exception:
                    traceback.print_exc()
                    from PIL import Image, ImageDraw
                    img = Image.new("RGB", (512, 512), color=(220, 150, 150))
                    d = ImageDraw.Draw(img)
                    d.text((20, 20), "Erro geração sem LLM", fill=(0, 0, 0))

                end = time.time()
                elapsed = round(end - start, 2)
                filename = f"sem_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                path = os.path.join(STATIC_DIR, "outputs_sem_llm", filename)
                img.save(path)

                SESSAO["imagens_sem_llm"].append(f"/static/outputs_sem_llm/{filename}")
                SESSAO["tempos_sem_llm"].append(elapsed)

                steps_done += 1
                with progress_lock:
                    SESSAO["progress"] = int((steps_done / total_steps) * 100)

        # ---------------- COM LLM ----------------
        if modo_geracao in ("com_llm", "ambos"):
            # otimiza prompt com LLM (1 step)
            prompt_otimizado = ""
            try:
                if pipe_com is not None and hasattr(pipe_com, "melhorar_prompt"):
                    prompt_otimizado = pipe_com.melhorar_prompt(f"{user_text}. A atmosfera e o tom emocional são de {', '.join(emocoes_relevantes)}.")
                elif pipe_com is not None and hasattr(pipe_com, "llm_generate"):
                    prompt_otimizado = pipe_com.llm_generate(f"Melhore o seguinte prompt para geração de imagem, sem inventar nada além do necessário: {user_text}")
                else:
                    # fallback: apenas reusa o texto concatenado
                    prompt_otimizado = f"{user_text}. A atmosfera e o tom emocional são de {', '.join(emocoes_relevantes)}."
            except Exception:
                traceback.print_exc()
                prompt_otimizado = f"{user_text}. A atmosfera e o tom emocional são de {', '.join(emocoes_relevantes)}."

            SESSAO["prompt_otimizado"] = prompt_otimizado
            steps_done += 1
            with progress_lock:
                SESSAO["progress"] = int((steps_done / total_steps) * 100)

            for i in range(num_imgs):
                start = time.time()
                try:
                    if pipe_com is not None and hasattr(pipe_com, "gerar_imagem_com_llm"):
                        _prompt, imgs = pipe_com.gerar_imagem_com_llm(user_text, emocoes_relevantes, num_img=1)
                        img = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
                    else:
                        from PIL import Image, ImageDraw
                        img = Image.new("RGB", (512, 512), color=(200, 200, 230))
                        d = ImageDraw.Draw(img)
                        d.text((20, 20), f"COM LLM\n{i}", fill=(0, 0, 0))
                except Exception:
                    traceback.print_exc()
                    from PIL import Image, ImageDraw
                    img = Image.new("RGB", (512, 512), color=(150, 200, 150))
                    d = ImageDraw.Draw(img)
                    d.text((20, 20), "Erro geração com LLM", fill=(0, 0, 0))

                end = time.time()
                elapsed = round(end - start, 2)
                filename = f"com_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                path = os.path.join(STATIC_DIR, "outputs_com_llm", filename)
                img.save(path)

                SESSAO["imagens_com_llm"].append(f"/static/outputs_com_llm/{filename}")
                SESSAO["tempos_com_llm"].append(elapsed)

                steps_done += 1
                with progress_lock:
                    SESSAO["progress"] = int((steps_done / total_steps) * 100)

        # finaliza
        with progress_lock:
            SESSAO["progress"] = 100
            SESSAO["in_progress"] = False

    except Exception as e:
        traceback.print_exc()
        with progress_lock:
            SESSAO["erro_geracao"] = str(e)
            SESSAO["in_progress"] = False
            SESSAO["progress"] = 100


# ===========================================================
# ETAPA 2 → GERAR IMAGENS (inicia background + retorna rapidamente)
# ===========================================================
@app.post("/gerar_imagens")
async def gerar_imagens(
    request: Request,
    modo_geracao_hidden: str = Form("ambos"),
    modo_acoes_hidden: str = Form("manual")
):
    # Garante que o usuário já processou emoções antes
    if not SESSAO["texto"]:
        return RedirectResponse("/", status_code=303)

    # Se já estiver rodando, retorna indicando isso
    with progress_lock:
        if SESSAO["in_progress"]:
            return JSONResponse({"started": False, "reason": "already_running"})

        # inicia thread de geração
        modo = modo_geracao_hidden
        t = threading.Thread(target=_gera_imagens_background, args=(modo, 3), daemon=True)
        t.start()

    # Se a requisição vier do formulário tradicional (não AJAX), redirecionar para a página (o JS na página começará a polular progresso)
    # Aqui detectamos por header simple: se é fetch X-Requested-With, mas formulário normal não envia; retornamos template.
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_text": SESSAO["texto"],
            "texto_completo": SESSAO["texto_completo"],
            "emocoes": SESSAO["emocoes_previstas"],
            "modo_geracao": modo_geracao_hidden,
            "modo_acoes": modo_acoes_hidden,
            "mostrar_progresso": True
        }
    )


# ===========================================================
# Endpoint polled pelo frontend para saber o progresso
# ===========================================================
@app.get("/progress")
async def get_progress():
    with progress_lock:
        return {
            "in_progress": SESSAO["in_progress"],
            "progress": SESSAO["progress"],
            "erro": SESSAO.get("erro_geracao")
        }


# ===========================================================
# Endpoint para retornar resultado final (URLs e tempos)
# ===========================================================
@app.get("/generation_result")
async def generation_result():
    return {
        "imagens_sem_llm": SESSAO["imagens_sem_llm"],
        "tempos_sem_llm": SESSAO["tempos_sem_llm"],
        "imagens_com_llm": SESSAO["imagens_com_llm"],
        "tempos_com_llm": SESSAO["tempos_com_llm"],
        "prompt_otimizado": SESSAO.get("prompt_otimizado", ""),
        "erro": SESSAO.get("erro_geracao")
    }
