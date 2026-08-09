variable "aws_region" {
  description = "Região AWS para provisionamento"
  type        = string
  default     = "us-east-1"
}

variable "service_name" {
  description = "Nome do serviço de inferência"
  type        = string
  default     = "diag-opt-inference"
}

variable "image_tag" {
  description = "Tag da imagem do container no ECR"
  type        = string
  default     = "latest"
}

variable "container_port" {
  description = "Porta exposta pelo container do serviço"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Unidades de CPU da tarefa Fargate (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memória (MiB) da tarefa Fargate"
  type        = number
  default     = 1024
}

variable "min_capacity" {
  description = "Número mínimo de tarefas (auto-scaling)"
  type        = number
  default     = 2
}

variable "max_capacity" {
  description = "Número máximo de tarefas (auto-scaling)"
  type        = number
  default     = 10
}

variable "cpu_target_utilization" {
  description = "Utilização de CPU alvo (%) para o auto-scaling"
  type        = number
  default     = 70
}

variable "requests_per_target" {
  description = "Requisições por tarefa alvo para o auto-scaling"
  type        = number
  default     = 1000
}

variable "vpc_id" {
  description = "VPC onde os recursos serão criados"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnets para o serviço e o load balancer"
  type        = list(string)
  default     = []
}
