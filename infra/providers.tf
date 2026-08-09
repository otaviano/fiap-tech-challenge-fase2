terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project = "tech-challenge-fase2"
      Purpose = "diagnostico-otimizacao-ml"
      Managed = "terraform"
    }
  }
}
