# Sistema de chat usando RAG, LangGraph e FastAPI

Sistema de atendimento e suporte ao cliente automatizado desenvolvido com **FastAPI**, **LangGraph** e **Google Gemini**. O projeto responde a dúvidas com base em políticas pré-estabelecidas da empresa (como horário de funcionamento, regras de suporte e preços de produtos) utilizando busca vetorial RAG.

Arquitetura e Diferenciais Técnicos

#### 1.  Nó de Segurança e Criptografia (LGPD Compliance)
* Intercepta todas as mensagens antes de chegarem ao LLM.
* Identifica automaticamente dados sensíveis (como CPFs) via expressões regulares.
* Criptografa os dados utilizando **Fernet (cryptography)** e mascara o texto na conversa, impedindo o envio indevido de dados confidenciais para APIs externas.

#### 2.  Nó de Triagem Dinâmica (Otimização de Custos)
* Analisa a intenção da mensagem e o histórico da conversa.
* **Modelo Econômico (`gemini-3.5-flash-lite`):** Utilizado para saudações, conversas gerais e consultas simples ao catálogo de produtos com chamadas de ferramentas (*Function Calling*).
* **Modelo Avançado (`gemini-3.8-flash`):** Acionado dinamicamente para consultas complexas sobre políticas de suporte e regras de negócio.
* **Resultado:** Redução expressiva no custo por requisição sem perda da qualidade nas respostas complexas.

#### 3.  RAG & Ferramentas de Consulta
* **FAISS (Vector Store):** Busca de similaridade semântica para recuperar regras e políticas da empresa em tempo real.
* **Custom Tools:** Ferramentas dedicadas para conversão de moedas e busca de produtos por nome/código no banco de dados.

#### 4.  Memória Persistente por Sessão
* Utiliza o `MemorySaver` do LangGraph para manter o contexto das conversas isolado por `chat_id`.