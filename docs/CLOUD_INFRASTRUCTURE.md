# Enterprise Cloud Infrastructure & Deployment Blueprint

This document details the production cloud deployment specification for **AegisRAG** on **Amazon Web Services (AWS)** and **Qdrant Cloud**.

---

## 1. Cloud Architecture Topology

```
                                   +--------------------------------+
                                   |         Internet Users         |
                                   +---------------+----------------+
                                                   |
                                                   v
                                   +---------------+----------------+
                                   |      AWS CloudFront (CDN)      |
                                   |  SSL / Edge Caching / WAF      |
                                   +-------+----------------+-------+
                                           |                |
                     +---------------------+                +---------------------+
                     | (Static Web Assets)                  | (Dynamic API Calls)
                     v                                      v
       +-------------+-------------+          +-------------+-------------+
       |       AWS S3 Bucket       |          |     AWS API Gateway       |
       | Streamlit Frontend Build  |          | Rate Limiting & Auth      |
       +---------------------------+          +-------------+-------------+
                                                            |
                                                            v
                                              +-------------+-------------+
                                              | Application Load Balancer |
                                              +-------------+-------------+
                                                            |
                                      +---------------------+---------------------+
                                      |                                           |
                                      v                                           v
                        +-------------+-------------+               +-------------+-------------+
                        |  AWS ECS Fargate Task #1  |               |  AWS ECS Fargate Task #2  |
                        |  - FastAPI Container      |               |  - FastAPI Container      |
                        |  - In-Memory BM25 Index   |               |  - In-Memory BM25 Index   |
                        +-------------+-------------+               +-------------+-------------+
                                      |                                           |
                                      +---------------------+---------------------+
                                                            |
                                      +---------------------+---------------------+
                                      |                                           |
                                      v                                           v
                        +-------------+-------------+               +-------------+-------------+
                        |       Qdrant Cloud        |               |   Groq LPU Inference API  |
                        | - HNSW Dense Vector Index |               | - Low-latency Llama-3.3   |
                        | - Multi-AZ Cluster        |               | - Dedicated LPU Cluster   |
                        +---------------------------+               +---------------------------+
```

---

## 2. Infrastructure as Code (Terraform Manifest)

Below is the production Terraform configuration for provisioning the containerized ECS Fargate cluster:

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# ECS Cluster
resource "aws_ecs_cluster" "aegis_cluster" {
  name = "aegis-rag-production"
}

# ECS Task Definition
resource "aws_ecs_task_definition" "aegis_task" {
  family                   = "aegis-rag-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024" # 1 vCPU
  memory                   = "2048" # 2 GB RAM

  container_definitions = jsonencode([
    {
      name      = "aegis-rag-service"
      image     = "your-account-id.dkr.ecr.us-east-1.amazonaws.com/aegis-rag:latest"
      essential = true
      portMappings = [
        { containerPort = 8000, hostPort = 8000 },
        { containerPort = 8501, hostPort = 8501 }
      ]
      environment = [
        { name = "ENVIRONMENT", value = "production" },
        { name = "GROQ_MODEL", value = "qwen/qwen3.8-27b" }
      ]
      secrets = [
        {
          name      = "GROQ_API_KEY"
          valueFrom = "arn:aws:secretsmanager:us-east-1:123456789012:secret:groq_api_key"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/aegis-rag"
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECS Service with Autoscaling
resource "aws_ecs_service" "aegis_service" {
  name            = "aegis-rag-ecs-service"
  cluster         = aws_ecs_cluster.aegis_cluster.id
  task_definition = aws_ecs_task_definition.aegis_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = ["subnet-abc12345", "subnet-def67890"]
    security_groups  = ["sg-0123456789abcdef0"]
    assign_public_ip = true
  }
}
```

---

## 3. Observability & Telemetry (OpenTelemetry + CloudWatch)

1. **Distributed Tracing:** Traces are tagged with `trace_id` propagated across each node in the LangGraph CRAG state machine.
2. **Token & Latency Metrics:** Every API call emits:
   - `llm_time_to_first_token_ms`
   - `llm_tokens_per_second`
   - `retrieval_dense_latency_ms`
   - `retrieval_bm25_latency_ms`
   - `rrf_fusion_latency_ms`
   - `hallucination_score`
3. **Automated Alarm Triggers:** CloudWatch alarms trigger automated container scale-out if P95 latency exceeds $2,500\text{ ms}$ or if ungrounded hallucination rate exceeds $5\%$.
