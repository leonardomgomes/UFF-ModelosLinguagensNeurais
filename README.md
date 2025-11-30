Segue a explicação de cada arquivo:</br>
* formata_traduz_go_emotions.py - Baixa, traduz para português, formata representação do atributo label para One-Hot e armazena no computador o dataset GoEmotions em Português.
* treina_emocoes_go_emotions.py - realiza o treinamento de modelo pré-treinado transformer para reconhecimento de emoções em texto em Português.
* index.html: monta a estrutura de objetos html do sistema na web.
* main.py: orquestra a execução do sistema, chamando o FastAPI que é um framework web Python.
* emotion\_model.py: executa o modelo Transformer, treinado por fine-tuning, para previsão de emoções.
* pipeline\_sem\_llm.py: executa a geração de imagens a partir do texto do usuário + emoções.
* pipeline\_com\_llm.py: executa a melhoria do texto de prompt a partir do texto do usuário + emoções e em seguida executa a geração de imagens.
* Anexo I – Evidências dos Testes: contendo as evidências dos 9 testes executados para o sistema.
