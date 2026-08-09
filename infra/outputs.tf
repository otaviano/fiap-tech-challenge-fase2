output "alb_dns_name" {
  description = "DNS público do Application Load Balancer"
  value       = aws_lb.service.dns_name
}

output "ecr_repository_url" {
  description = "URL do repositório ECR para push da imagem"
  value       = aws_ecr_repository.service.repository_url
}

output "ecs_cluster_name" {
  description = "Nome do cluster ECS"
  value       = aws_ecs_cluster.this.name
}

output "log_group" {
  description = "Grupo de logs no CloudWatch"
  value       = aws_cloudwatch_log_group.service.name
}
