from datasets import Dataset, load_from_disk
from datasets import Features, Value, Sequence
from datetime import datetime
import os
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np
import json
import csv

emocoes =   ['admiração', 'diversão', 'raiva', 'irritação', 'aprovação', 'carinho', 'confusão', 'curiosidade', 'desejo', 'decepção', 
            'desaprovação', 'nojo', 'constrangimento', 'empolgação', 'medo', 'gratidão', 'pesar', 'alegria', 'amor', 'nervosismo', 'otimismo', 
            'orgulho', 'percepção', 'alívio', 'remorso', 'tristeza', 'surpresa', 'neutro']

emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

dict_emocoes = {
0: 'admiração', 1: 'diversão', 2: 'raiva', 3: 'irritação', 4: 'aprovação', 5: 'carinho', 6: 'confusão', 7: 'curiosidade', 8: 'desejo',
9: 'decepção', 10: 'desaprovação', 11: 'nojo', 12: 'constrangimento', 13: 'empolgação', 14: 'medo', 15: 'gratidão', 16: 'pesar',
17: 'alegria', 18: 'amor', 19: 'nervosismo', 20: 'otimismo', 21: 'orgulho', 22: 'percepção', 23: 'alívio', 24: 'remorso', 25: 'tristeza', 
26: 'surpresa', 27: 'neutro'}

dict_emotions = {
0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire',
9: 'disappointment', 10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief',
17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 
26: 'surprise', 27: 'neutral'}

dict_grupo_emocoes = {
'positivo': [0, 1, 4, 5, 8, 13, 15, 17, 18, 20, 21, 23],
'negativo': [2, 3, 9, 10, 11, 12, 14, 16, 19, 24, 25],
'ambíguo': [6, 7, 22, 26],
'neutro': [27]
}

dict_tradu = {'positivo':'positive', 'negativo':'negative', 'ambíguo':'ambiguous', 'neutro':'neutral'}

#########################################################################################################################

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# 1️⃣ Carregar dataset 
# Dataset já carregado da internet através do script 'formata_traduz_go_emotions.py' onde foi também executado: armazenamento no computador o dataset
# "google-research-datasets/go_emotions", realizada a formatação das labels para multi-classificação de 28 emoções e 
# a tradução do inglês para português usando LLM "Helsinki-NLP/opus-mt-tc-big-en-pt"
dset_test = Dataset.load_from_disk("d:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/goemotions_ptbr/test")
dset_validation = Dataset.load_from_disk("d:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/goemotions_ptbr/validation")
dset_train = Dataset.load_from_disk("d:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/goemotions_ptbr/train")
print(dset_test)
print(dset_test[0])
print(dset_test[0]["labels"][:5])
print(type(dset_test[0]["labels"][0]))
print('\n')
print(dset_validation)
print(dset_validation[0])
print(dset_validation[0]["labels"][:5])
print(type(dset_validation[0]["labels"][0]))
print('\n')
print(dset_train)
print(dset_train[0])
print(dset_train[0]["labels"][:5])
print(type(dset_train[0]["labels"][0]))
print('\n')

'''
features_float = Features({
    "text": Value("string"),
    "labels": Sequence(Value("float32")),
    "id": Value("string"),
    "labels_originais": Sequence(Value("int32")),
    "text_original": Value("string"),
})

# Agora sim converte e redefine o tipo:
dset_train = dset_train.map(lambda e: {"labels": [float(x) for x in e["labels"]]}, features=features_float)
dset_validation = dset_validation.map(lambda e: {"labels": [float(x) for x in e["labels"]]}, features=features_float)
dset_test = dset_test.map(lambda e: {"labels": [float(x) for x in e["labels"]]}, features=features_float)
'''

# Ajusta o tipo apenas do campo "labels" para cada conjunto

features_train = dset_train.features.copy()
features_train["labels"] = Sequence(Value("float32"))

features_val = dset_validation.features.copy()
features_val["labels"] = Sequence(Value("float32"))

features_test = dset_test.features.copy()
features_test["labels"] = Sequence(Value("float32"))

# Aplica a conversão mantendo o restante do schema intacto
dset_train = dset_train.map(lambda e: {"labels": [float(x) for x in e["labels"]]}, features=features_train)
dset_validation = dset_validation.map(lambda e: {"labels": [float(x) for x in e["labels"]]}, features=features_val)
dset_test = dset_test.map(lambda e: {"labels": [float(x) for x in e["labels"]]}, features=features_test)

print(dset_test[0])
print(dset_test[0]["labels"][:5])
print(type(dset_test[0]["labels"][0]))
print('\n')
print(dset_validation[0])
print(dset_validation[0]["labels"][:5])
print(type(dset_validation[0]["labels"][0]))
print('\n')
print(dset_train[0])
print(dset_train[0]["labels"][:5])
print(type(dset_train[0]["labels"][0]))

# Inicializa variáveis
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
modelo = "microsoft/mdeberta-v3-base"
output_dir = f'd:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/{modelo}/{timestamp}'
qtd_emocoes = len(emocoes)

# 2️⃣ Criar tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelo)

# 3️⃣ Tokenizar dataset
def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, max_length=64, padding=False)

# tokenized_ds = dset_traduz.map(tokenize_function, batched=True)
tokenized_ds_train = dset_train.map(tokenize_function, batched=True)
tokenized_ds_valid = dset_validation.map(tokenize_function, batched=True)
tokenized_ds_test = dset_test.map(tokenize_function, batched=True)

# Antes de passar os dados ao Trainer, converta os labels para float no dataset tokenizado.
# -----------------------------------------------------------
# CORREÇÃO: converter os labels para float (necessário para BCEWithLogitsLoss).  O DebertaV2ForSequenceClassification usa automaticamente a loss binária (BCEWithLogitsLoss) quando o número de labels > 1, e essa função espera tensores float entre 0.0 e 1.0.
# -----------------------------------------------------------
def cast_labels_to_float(batch):
    return {"labels": [float(x) for x in batch["labels"]]}

# tokenized_ds_train = tokenized_ds_train.map(cast_labels_to_float)
# tokenized_ds_valid = tokenized_ds_valid.map(cast_labels_to_float)
# tokenized_ds_test  = tokenized_ds_test.map(cast_labels_to_float)
# -----------------------------------------------------------


# print(tokenized_ds)
# print(f'tokenized_ds: {tokenized_ds[0]}\n')

print(tokenized_ds_train)
print(f'tokenized_ds_train: {tokenized_ds_train[0]}\n')
print(tokenized_ds_valid)
print(f'tokenized_ds_valid: {tokenized_ds_valid[0]}\n')
print(tokenized_ds_test)
print(f'tokenized_ds_test: {tokenized_ds_test[0]}\n')

# 4️⃣ Criar data collator (padding dinâmico) - evita truncar os textos de maneira rígida e tamanho fixo para preencher os paddings zerados nas posições vazias dos textos menores.  
# Usando 'DataCollatorWithPadding' cria tamanhos de textos variados, usando paddings apenas necessários, poupando assim a memória.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# print(data_collator)

# 5️⃣ Definir modelo e args
model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=qtd_emocoes)
model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=qtd_emocoes, problem_type="multi_label_classification")
model = model.to(torch.device("cpu"))

training_args = TrainingArguments(
    output_dir=output_dir,  # "./results",
    num_train_epochs=3,
    fp16=False,
    eval_strategy='epoch',
    save_strategy='epoch',      # <- salva ao final de cada época
    save_total_limit=1,         # <- mantém apenas o último checkpoint, mantém só o melhor checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    load_best_model_at_end=True,    # <- ESSENCIAL  # salva o melhor modelo baseado na métrica
    metric_for_best_model="f1",     # <- métrica usada para decidir o melhor
    greater_is_better=True,         # <- se métrica maior = melhor (ex: f1, accuracy)
    logging_dir=f"{output_dir}/logs",        # <- salva logs no diretório
    # logging_strategy='epoch',                # <- mostra métricas no terminal
    logging_strategy="steps",
    logging_steps=100,
    logging_first_step=True,
    report_to=["tensorboard"],                      # evita wandb se não estiver usando
)

# 6️⃣ Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=tokenized_ds["train"],
    # eval_dataset=tokenized_ds["validation"],
    train_dataset= tokenized_ds_train.shuffle(seed=42),
    eval_dataset= tokenized_ds_valid,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# === TREINAMENTO ===
train_result = trainer.train()
trainer.save_model(f"{output_dir}/best_model")
train_metrics = train_result.metrics
train_metrics["train_samples"] = len(tokenized_ds_train)

# === VALIDAÇÃO FINAL ===
# A validação é feita automaticamente durante o treino (por epoch)
# Aqui fazemos uma última avaliação explícita no conjunto de validação
valid_metrics = trainer.evaluate(eval_dataset=tokenized_ds_valid)
valid_metrics["validation_samples"] = len(tokenized_ds_valid)

# === TESTE FINAL ===
# Usa o melhor modelo salvo automaticamente
test_metrics = trainer.evaluate(eval_dataset=tokenized_ds_test)
test_metrics["test_samples"] = len(tokenized_ds_test)

# Função auxiliar para salvar em 3 formatos
def salvar_metricas(nome, metricas):
    # JSON
    with open(f"{output_dir}/{nome}.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    # TXT
    with open(f"{output_dir}/{nome}.txt", "w", encoding="utf-8") as f:
        for k, v in metricas.items():
            f.write(f"{k}: {v}\n")
    # CSV
    with open(f"{output_dir}/{nome}.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["métrica", "valor"])
        for k, v in metricas.items():
            writer.writerow([k, v])

# Salvar todas as fases
salvar_metricas("train_metrics", train_metrics)
salvar_metricas("validation_metrics", valid_metrics)
salvar_metricas("test_metrics", test_metrics)

# Print no console para acompanhamento
print("\n📊 Resultados Finais:")
for fase, metricas in [("Treino", train_metrics), ("Validação", valid_metrics), ("Teste", test_metrics)]:
    print(f"\n--- {fase} ---")
    for k, v in metricas.items():
        print(f"{k}: {v}")
