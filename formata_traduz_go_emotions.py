from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import MarianMTModel, MarianTokenizer, M2M100Tokenizer, M2M100ForConditionalGeneration
from datetime import datetime
import os
import torch

emocoes =   ['admiração', 'diversão', 'raiva', 'irritação', 'aprovação', 'carinho', 'confusão', 'curiosidade', 'desejo', 'decepção', 
            'desaprovação', 'nojo', 'constrangimento', 'empolgação', 'medo', 'gratidão', 'tristeza', 'alegria', 'amor', 'nervosismo', 'otimismo', 
            'orgulho', 'percepção', 'alívio', 'remorso', 'tristeza', 'surpresa', 'neutro']

qtd_emocoes = len(emocoes)  # 27 emoções (12 positivas, 11 negativas e 4 ambíguas) + 1 neutro = 28 classificações

# Função que formata as labels para multi-classificação atribuindo um novo conteúdo de 28 posições (27 emoções + 1 emoção neutra) com 
# marcação binária 1 nas emoções existentes na sentença e 0 para as emoções não existentes.  
# Foi criado um novo atributo 'labels_originais' que recebe o conteúdo do atributo original 'labels'.  Em 'labels' são armazenadas as numerações
# das emoções que são números de 0 a 27 e pode haver mais de um número se houver mais de uma emoção no texto.
# Armazena em 'labels' o conteúdo das emoções do texto em formato para multi-classificação binária de 28 posições. 
def formata_labels(dado):
    labels = [0.0] * qtd_emocoes
    dado['labels_originais'] = dado['labels']
    for emot in range(qtd_emocoes):
        if (emot in dado['labels_originais']):
            labels[emot] = float(1.0)
    dado['labels'] = labels
    return dado

# 1️⃣ Carregar dataset (modo streaming = False) Dataset Go_Emotions
ds = load_dataset(
    "google-research-datasets/go_emotions",
    "simplified",              # ou "raw" (dependendo da configuração desejada)
    streaming=False     # sem o 'streaming=True' o dataset torna-se um dataset regular, com 'streaming=True' o dataset é um IterableDataset e não é visível os dados e nem quantidade em um simples print: https://chatgpt.com/g/g-p-690e5dea4b7c8191a29e11c185416a7a-doutorado-uff-2025-semestre-2-mod-linguagem/c/690e9ef1-e448-832b-8c34-e099f351db08
)

# Separa o dataset em conjuntos de treino, validação e teste
ds_train = ds['train']
ds_valid = ds['validation']
ds_test = ds['test']
print(f'ds_train: \n{ds_train}')
print(f'ds_valid: \n{ds_valid}')
print(f'ds_test: \n{ds_test}')

# Realiza a formatação das labels criando novo atributo do Dataset, armazenando em formato multi-classificação e marcando a(s) emoção(ôes) do texto
dset_train = ds_train.map(formata_labels)
dset_valid = ds_valid.map(formata_labels)
dset_test = ds_test.map(formata_labels)
print(f'dset_train: \n{dset_train}')
print(f'dset_valid: \n{dset_valid}')
print(f'dset_test: \n{dset_test}')

# Modelo Tradutor
model_name_translate = "Helsinki-NLP/opus-mt-tc-big-en-pt"    # modelo tradutor muito pesado

# Aplicando o Tokenizador para o modelo do tradutor
trans_tokenizer = MarianTokenizer.from_pretrained(model_name_translate)
trans_model = MarianMTModel.from_pretrained(model_name_translate)

# Função de tradução - Executa o modelo tradutor.  Será chamada pela função 'map' para ser aplicada individualmente em cada agrupamento do dataset.
# Criação de novo atributo 'text_original' que recebe o conteúdo do atributo original 'text' que é o texto em inglês
# Armazenamento da tradução do texto de inglês para português no atributo 'text'
def traduzir_batch(batch):
    textos = batch["text"]
    batch["text_original"] = batch["text"]
    # Traduz em lotes (para eficiência)
    inputs = trans_tokenizer(textos, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = trans_model.generate(**inputs)
    traduzidos = trans_tokenizer.batch_decode(translated, skip_special_tokens=True)
    batch["text"] = traduzidos
    return batch


# 3️⃣ Traduzir dataset inteiro (streaming desativado), traduzindo cada agrupamento: teste (5.427 registros), validação (5.426 registros) e 
# treino (43.410 registros).  Após a tradução armazena no computador.
# A tradução levou: teste (5427/5427 [1:22:21<00:00,  1.10 examples/s]), validação (5426/5426 [1:35:09<00:00,  1.05s/ examples]) e treino (43410/43410 [13:40:49<00:00,  1.13s/ examples])
print("Traduzindo dataset para PT-BR...")

print("Iniciando Tradução para Teste!")
dset_traduz_test = dset_test.map(traduzir_batch, batched=True, batch_size=16)   # Tradução por batch para ser mais rápido e poupar memória
print("Tradução concluída para Teste!")
print(f'dset_traduz_test: \n{dset_traduz_test}')
print(f'{dset_traduz_test[1]}\n')
dset_traduz_test.save_to_disk("d:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/goemotions_ptbr/test")   # Salvar a versão traduzida

print("Iniciando Tradução para Validação!")
dset_traduz_valid = dset_valid.map(traduzir_batch, batched=True, batch_size=16)
print("Tradução concluída para Validação!")
print(f'dset_traduz_valid: \n{dset_traduz_valid}')
print(f'{dset_traduz_valid[1]}\n')
dset_traduz_valid.save_to_disk("d:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/goemotions_ptbr/validation")   # Salvar a versão traduzida

print("Iniciando Tradução para Treino!")
dset_traduz_train = dset_train.map(traduzir_batch, batched=True, batch_size=16)
print("Tradução concluída para Treino!")
print(f'dset_traduz_train: \n{dset_traduz_train}')
print(f'{dset_traduz_train[1]}\n')
dset_traduz_train.save_to_disk("d:/python3/doutorado-2-2025-periodo2/mod_ling_neural/go_emotion_model/goemotions_ptbr/train")   # Salvar a versão traduzida

'''
Formato do Dataset e exemplo
Formato do Dataset de teste (como exemplo)
Dataset({
    features: ['text', 'labels', 'id', 'labels_originais', 'text_original'],
    num_rows: 5427
})

Formato do dado armazenado (como exemplo)
0 - primeira sentença do agrupamento (índice 0)
{'text': 'Sinto muito pela sua situação :( Embora eu ame os nomes Sapphira, Cirilla e Scarlett!', 
'labels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'id': 'eecwqtt', 'labels_originais': [25], 
'text_original': 'I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!'}

Observação:
1 - foram criados os novos atributos 'text_original' e 'labels_originais';
2 - o conteúdo do atributo 'text' (texto original em inglês) foi armazenado no novo atributo 'text_original';
3 - o atributo 'text' recebe então a tradução em português do texto em inglês.
4 - o conteúdo do atributo 'labels' (vetor contendo identificadores numéricos das emoções existentes no texto, podendo ser números de 0 a 27)
    foi armazenado no novo atributo 'labels_original';
3 - o atributo 'telabelsxt' recebe então um novo vetor com marcação binária para multi-classificação sinalizando as emoções existentes no texto.
'''
