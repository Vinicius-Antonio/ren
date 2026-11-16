```markdown
# AGENTS.md — Projeto Ren (Assistente de Voz em Tempo Real)

Este documento define a especificação técnica, regras arquiteturais, convenções de código e critérios de aceite para a implementação da assistente de voz **Ren** (anteriormente codinome Hermes). Qualquer agente autônomo de codificação deve seguir rigorosamente as restrições e diretrizes aqui descritas.

---

## 1. Visão Geral e Objetivo

Construir uma aplicação local em Python que atua como uma assistente de voz inteligente de baixa latência chamada **Ren**.
O agente deve:
1. Permanecer em **Standby de Custo Zero**, consumindo apenas CPU local para escutar uma Wake Word ("Ren" ou "Hermes").
2. Ao ser chamada, transicionar para o estado **Ativo**, conectando-se via WebSocket à **Gemini Multimodal Live API**.
3. Realizar streaming bidirecional de áudio ponta a ponta (envia fala em PCM 16kHz e toca síntese nativa em PCM 24kHz).
4. Utilizar **Grounding com Google Search** nativo na sessão para responder dúvidas factuais e em tempo real.
5. Encerrar a sessão e retornar ao standby automaticamente via **Detecção de Silêncio Local (VAD)** ou por comando explícito de despedida do usuário.

---

## 2. Pilha Tecnológica (Tech Stack)

- **Linguagem:** Python 3.10+ (Digitação estrita com `typing` / PEP 484).
- **SDK do Gemini:** `google-genai` (utilizar estritamente `client.aio.live.connect`).
- **Engine de Wake Word:** `pvporcupine` (Picovoice Porcupine) ou `openwakeword`.
- **Captura e Reprodução de Áudio:** `PyAudio` (com bindings `PortAudio`).
- **Processamento Numérico de Áudio:** `NumPy` (para cálculo de RMS e VAD).
- **Gerenciamento de Ambiente:** `python-dotenv`.
- **Modelo LLM Alvo:** `gemini-2.0-flash-exp` (ou o endpoint multimodal live estável equivalente).

---

## 3. Especificações Técnicas de Áudio

- **Formato Geral:** PCM 16-bit signed integer (`paInt16`).
- **Canais:** Mono (1 canal).
- **Entrada (Microfone -> Gemini):** Taxa de amostragem de `16.000 Hz`, chunk de `1024` frames.
- **Saída (Gemini -> Alto-falante):** Taxa de amostragem de `24.000 Hz`.
- **MIME Type enviado:** `audio/pcm;rate=16000`.

---

## 4. Máquina de Estados e Ciclo de Vida

O agente opera exclusivamente em 2 estados formais:


```

┌────────────────────────────────────────────────────────┐
│                   ESTADO 1: STANDBY                    │
│ - Microfone aberto em buffer local (Porcupine).        │
│ - Zero conexões ativas na internet. Zero gasto de API. │
└──────────────────────────┬─────────────────────────────┘
│ Wake Word ("Ren" / "Hermes") detectada
▼
┌────────────────────────────────────────────────────────┐
│                    ESTADO 2: ATIVO                     │
│ - Abre WebSocket (LiveConnectConfig).                  │
│ - Task A: Gravação contínua e envio de chunks.        │
│ - Task B: Recepção contínua e playback de áudio.       │
│ - Task C: Monitor VAD (RMS de silêncio > 6s).          │
└──────────────────────────┬─────────────────────────────┘
│ Silêncio atingido OU comando de despedida
▼
Retorna ao ESTADO 1

```

---

## 5. Estrutura do Repositório

```text
ren-agent/
├── AGENTS.md                  # Especificação do projeto (este arquivo)
├── .env.example               # Template de variáveis obrigatórias
├── .gitignore                 # Deve ignorar .env, venv/, *.ppn, *.onnx
├── requirements.txt           # Dependências com versões fixadas/mínimas
├── run.py                     # Entry point do loop principal
├── models/
│   └── ren.ppn                # Arquivo binário da Wake Word (se usar Porcupine)
└── src/
    ├── __init__.py
    ├── config.py              # Leitura de variáveis e system prompt
    ├── audio/
    │   ├── __init__.py
    │   ├── vad.py             # Algoritmo de cálculo de RMS e detecção de silêncio
    │   └── stream_manager.py  # Abertura segura e liberação de streams PyAudio
    ├── detector/
    │   ├── __init__.py
    │   └── wake_word.py       # Wrapper de escuta e processamento da Wake Word
    └── gemini/
        ├── __init__.py
        ├── session.py         # Sessão assíncrona Gemini Live (Send/Receive/Tools)
        └── prompts.py         # Instruções de sistema e personalidade da Ren

```

---

## 6. Regras de Implementação para o Agente de Código

1. **Gestão Segura de Recursos PyAudio:**
* Streams de áudio de entrada e saída devem **sempre** ser fechados (`stream.stop_stream()`, `stream.close()`) em blocos `finally:`.
* Evitar concorrência pelo mesmo device de gravação: feche ou pause o stream de Wake Word antes de abrir o stream do Gemini.


2. **Concorrência Assíncrona Não-Bloqueante:**
* Métodos I/O do PyAudio (`stream.read()`, `stream.write()`) são síncronos e bloqueantes. Sempre execute-os dentro de executores assíncronos:
```python
await loop.run_in_executor(None, stream.read, chunk_size, False)

```


* Use `asyncio.Event()` para sinalização de parada entre tarefas (`t_rec`, `t_play`, `t_vad`).


3. **Restrições de Prompt do Modelo (Audio-Only):**
* O `system_instruction` deve proibir estritamente Markdown (tabelas, títulos, bullets, asteriscos), já que o retorno é consumido como áudio puro.
* Forçar respostas conversacionais, diretas e com pontuação clara para cadência de fala natural.
* Habilitar `tools=[types.Tool(google_search=types.GoogleSearch())]`.


4. **Tratamento de Falhas e Resiliência:**
* Quedas de WebSocket (`exceptions.APIError`, `asyncio.CancelledError`, quedas de rede) devem ser capturadas sem derrubar o script principal, retornando graciosamente ao estado de Standby.



---

## 7. Roteiro de Tarefas para o Agente (Implementation Roadmap)

* [ ] **Tarefa 1: Configuração do Ambiente**
* Criar `requirements.txt` e `.env.example`.
* Configurar `.gitignore` para dados sensíveis e binários.


* [ ] **Tarefa 2: Módulo de Áudio e VAD (`src/audio/`)**
* Implementar cálculo de RMS via NumPy em `src/audio/vad.py`.
* Implementar classes utilitárias para iniciar streams de 16kHz e 24kHz em `src/audio/stream_manager.py`.


* [ ] **Tarefa 3: Detector de Wake Word (`src/detector/`)**
* Encapsular a inicialização e escuta do Porcupine/openWakeWord em `src/detector/wake_word.py`.


* [ ] **Tarefa 4: Cliente Gemini Live (`src/gemini/`)**
* Implementar conexão WebSocket com envio de blobs PCM e consumo de buffers de saída.
* Injetar ferramenta de busca e instruções de personalidade em `src/gemini/session.py`.


* [ ] **Tarefa 5: Orquestrador Principal (`run.py`)**
* Ligar a transição de estados Standby -> Ativo -> Standby em loop contínuo infinito com interrupção limpa por `KeyboardInterrupt`.



---

## 8. Critérios de Aceite (DoD - Definition of Done)

1. **Eficiência em Repouso:** O processo não realiza nenhuma requisição de rede enquanto aguarda a palavra-chave.
2. **Tempo de Resposta:** A transição do chamado "Ren" até a abertura da conexão deve ocorrer em menos de 1 segundo.
3. **Fluidez:** O áudio de saída não apresenta engasgos (*buffer underruns*) durante a fala do modelo.
4. **Desligamento Automático:** Ao ficar em silêncio por 6 segundos após uma resposta, a sessão é encerrada no terminal e o agente volta a exibir `[Ren] Em espera`.

```

```