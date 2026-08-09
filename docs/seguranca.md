# Segurança e Considerações de Produção

Este documento endereça explicitamente o ponto do feedback da Fase 1
(*"a análise de segurança e considerações sobre a implementação em produção
poderiam ter sido mais detalhadas"*).

## 1. Privacidade dos dados (LGPD)

- **LLM local por padrão**: nenhum dado de paciente é enviado a APIs externas.
  Toda a inferência e interpretação ocorrem dentro do perímetro da rede
  hospitalar — decisão central de arquitetura, não um detalhe.
- **Minimização e anonimização**: o contexto enviado ao LLM contém apenas
  características celulares agregadas e a predição, nunca identificadores do
  paciente. Os prompts trafegam dados anonimizados.
- **Retenção**: logs não devem conter PII; em produção, aplicar mascaramento e
  política de retenção compatível com a LGPD (art. 15/16).

## 2. Segurança da aplicação

| Risco | Mitigação |
|-------|-----------|
| Segredos no código | `.gitignore` bloqueia `.env`/`*.key`/`*.pem`; segredos via variáveis de ambiente / secret manager |
| Endpoint do LLM exposto | LLM em rede interna, sem exposição pública; autenticação/mTLS entre serviços |
| *Prompt injection* | Prompt com papel e restrições fixas; dados do paciente entram como conteúdo estruturado, não como instruções; saída validada por `llm/quality.py` |
| Entrada malformada na inferência | Validação de schema das features antes da predição |
| Dependências vulneráveis | Versões fixadas + varredura (`pip-audit`/Dependabot) no CI |
| Superfície de ataque do container | Imagem *slim*, usuário não-root, apenas portas necessárias |

## 3. Segurança clínica (uso responsável de IA)

- O sistema é **apoio à decisão**, nunca diagnóstico autônomo — reforçado no
  prompt e verificado automaticamente (`sem_afirmacao_categorica` em
  `quality.py`).
- **Priorização de recall da classe maligna**: o desenho da fitness minimiza
  falsos negativos, o erro de maior custo clínico.
- **Explicabilidade**: cada predição vem acompanhada das características que a
  embasaram (grounding), permitindo ao médico auditar o raciocínio.
- **Human-in-the-loop**: a palavra final é sempre do profissional de saúde.

## 4. Considerações de produção

- **Versionamento de modelo e dados**: artefatos versionados; rastreabilidade de
  qual modelo gerou qual predição (auditoria).
- **Governança de modelo**: registro de métricas de aprovação (recall mínimo)
  antes de promover um modelo otimizado ao ambiente produtivo.
- **Observabilidade**: alertas de *data drift*, de queda de recall e de aumento
  na taxa de *fallback* do LLM (ver [escalabilidade.md](escalabilidade.md)).
- **Disponibilidade**: o *fallback* determinístico garante que a interpretação
  nunca falhe totalmente se o LLM estiver indisponível — degradação graciosa.
- **Reprodutibilidade**: sementes fixas no GA e no split garantem resultados
  reproduzíveis para auditoria e revalidação.
