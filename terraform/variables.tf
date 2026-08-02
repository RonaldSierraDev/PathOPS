variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "pathml"
}

variable "container_port" {
  type    = number
  default = 8000
}

# 0.25 vCPU / 0.5GB -- the smallest Fargate task size, plenty for a ResNet18
# forward pass and keeps this near the project's own ~$10/mo cost cap.
variable "fargate_cpu" {
  type    = number
  default = 256
}

variable "fargate_memory" {
  type    = number
  default = 512
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Set to 0 to scale the API down to zero (e.g. between demo sessions) without tearing down the service."
}

variable "db_name" {
  type    = string
  default = "pathml"
}

variable "db_username" {
  type    = string
  default = "pathml"
}

# t4g (ARM/Graviton) is the cheapest RDS class and is the current free-tier
# eligible instance type for new AWS accounts (first 12 months).
variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "db_allocated_storage" {
  type    = number
  default = 20
}
