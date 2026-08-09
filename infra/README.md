# Infraestrutura como Código (Terraform)

Provisiona o serviço de **inferência + interpretação** do sistema de diagnóstico
na AWS, com **auto-scaling** para lidar com variações de demanda (requisito de
escalabilidade automática). É a materialização do diagrama em
[`../docs/escalabilidade.md`](../docs/escalabilidade.md).

> **Observação**: opção de nuvem do enunciado (pontuação extra). Os arquivos são
> um entregável válido de IaC. O `terraform apply` real exige uma conta AWS e
> credenciais — não incluídas por segurança.

## Recursos provisionados

- **ECR** — repositório da imagem do serviço.
- **ECS Fargate** — serviço *stateless* de inferência (sem servidores para
  gerenciar).
- **Application Load Balancer** — distribui carga entre as tarefas.
- **Application Auto Scaling** — *target tracking* por CPU (70%) e por número de
  requisições por tarefa, escalando de `min_capacity` a `max_capacity`.
- **CloudWatch Logs** — logging centralizado (tracking de desempenho).

## Uso

```bash
cd infra
terraform init
terraform plan  -var="image_tag=<tag>"
terraform apply -var="image_tag=<tag>"
```

## Estrutura

| Arquivo | Conteúdo |
|---------|----------|
| `providers.tf` | Provider AWS e versão do Terraform |
| `variables.tf` | Parâmetros (região, capacidades, CPU/memória) |
| `main.tf` | ECR, ECS, ALB, auto-scaling, logs |
| `outputs.tf` | URL do load balancer e nome do cluster |
