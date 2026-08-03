resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
}

# Required to make FARGATE_SPOT selectable by the service's capacity_provider_strategy below.
resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}-api"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  # The image tag is intentionally NOT part of this definition's inputs from
  # a variable -- `:latest` is pushed by scripts/../docker build+push step,
  # then `aws ecs update-service --force-new-deployment` picks it up. See
  # the repo README for the full push sequence.
  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:latest"
      essential = true
      portMappings = [{
        containerPort = var.container_port
        protocol      = "tcp"
      }]
      environment = [
        { name = "MODEL_S3_URI", value = "s3://${aws_s3_bucket.artifacts.bucket}/models/pcam_resnet18.pt" },
        { name = "MODEL_VERSION", value = "1" },
        { name = "PREDICTION_IMAGES_S3_BUCKET", value = aws_s3_bucket.artifacts.bucket },
        # Read-only inputs for /monitoring/drift; the threshold is shared with
        # the alarm so the console draws the line that actually fires.
        { name = "S3_ARTIFACTS_BUCKET", value = aws_s3_bucket.artifacts.bucket },
        { name = "CLOUDWATCH_NAMESPACE", value = var.cloudwatch_namespace },
        { name = "DRIFT_SHARE_THRESHOLD", value = tostring(var.drift_share_threshold) },
      ]
      secrets = [
        { name = "DATABASE_URL", valueFrom = aws_ssm_parameter.database_url.arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count

  # Fargate Spot instead of on-demand Fargate: interruptible with a 2-minute
  # warning, which is a real tradeoff (not for production SLAs) but cuts
  # compute cost substantially -- appropriate for a portfolio/demo service,
  # and keeps this within the project's own ~$10/mo budget rule.
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 100
  }

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs_service.id]
    assign_public_ip = true
  }

  # The first apply runs before any image exists in ECR, so the initial task
  # placement will fail to pull `:latest` -- expected, not an error. Push the
  # image (see README), then `aws ecs update-service --force-new-deployment`.
  wait_for_steady_state = false
}
