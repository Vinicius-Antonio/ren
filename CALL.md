# Ren no PC e na call

São dois processos independentes. O Ren do PC não conecta ao Discord. O Ren da call não abre o microfone nem o alto-falante do PC; recebe e reproduz áudio somente no canal de voz.

## Iniciar

- Duplo clique em `start-ren-call.cmd` para mandar o bot para a call com Gemini 3.8 Live, o modelo padrão. Esse modelo mantém as funções locais, mas desativa a busca nativa do Google porque a cota de busca para Gemini 3 no projeto gratuito está zerada.
- Para usar temporariamente o modelo anterior: `.\.venv\Scripts\python.exe run.py --mode call --model gemini-2.5-flash-native-audio-latest`.
- Duplo clique em `start-ren-pc.cmd` para iniciar o Ren local.
- PC: `.\.venv\Scripts\python.exe run.py --mode pc`
- Call em que você está (ou canal padrão): `.\.venv\Scripts\python.exe run.py --mode call`
- Canal específico: `.\.venv\Scripts\python.exe run.py --mode call --channel Geral`

O modo call precisa de `DISCORD_BOT_TOKEN` no `.env`. Configure também `DISCORD_USER_ID` com o ID numérico da sua conta para o bot entrar exatamente na sua call. Sem esse ID, ele procura um canal com pessoas ou usa o canal padrão. Depois fica aguardando “Ren” na chamada. Para sair da call e encerrar o processo, diga “Ren, sai da call” ou use Ctrl+C na janela correspondente. Se a conexão cair sem pedido de saída, o bot tenta voltar.

Você pode deixar o modo PC rodando e abrir outra janela para o modo call. Cada processo mantém sua própria sessão Gemini. Como o nome curto “Ren” não consta no vocabulário do Vosk instalado, o detector aceita aproximações fonéticas; a sensibilidade precisa ser validada em uma chamada real.

Qualquer amigo na mesma call pode chamar “Ren” e fazer uma pergunta. `DISCORD_USER_ID` serve apenas para escolher o canal de entrada do bot; não restringe quem pode falar com ele. Depois de uma resposta, há uma janela breve para perguntas de acompanhamento sem repetir o nome.

O bot identifica quem chamou “Ren” pelo ID do Discord antes de abrir a sessão Gemini. Durante esse turno, envia apenas a voz dessa pessoa ao modelo; falas simultâneas de outros membros não são atribuídas a ela. Uma chamada de outra pessoa durante a resposta fica pendente por até 10 segundos e é atendida depois que a Ren termina de falar. A janela de acompanhamento pertence somente à pessoa que iniciou o turno. O histórico registra o ID e o nome do autor; a Ren só associa alguém ao Vinícius quando o ID coincide com `DISCORD_USER_ID`.

O modo call requer o modelo Vosk local em `models/vosk-model-small-pt-0.3` para reconhecer a palavra “Ren” separadamente por participante. Se o modelo estiver ausente, o processo informa o erro em vez de atribuir uma voz misturada à pessoa errada.

Perfis dos amigos ficam em `data/friends.json` (formato em `data/friends.example.json`). Cada entrada contém `nome`, `fatos` e pelo menos um identificador: `discord_id` ou `discord_username`. O ID é preferível; o username serve como fallback exato. O Ren carrega os perfis ao iniciar; reinicie o processo da call após editar o arquivo.

Para publicar no chat, configure `DISCORD_TEXT_CHANNEL_ID` com o ID numérico de um canal de texto do mesmo servidor. O bot precisa poder ver e enviar mensagens nesse canal. Diga "Ren, responde e manda no chat" para publicar a transcrição depois da fala; pesquisas com fontes retornadas pela Gemini também publicam até cinco links. Sem o ID, o chat fica desativado e a voz continua funcionando.
