# Week 5: drift + error-rate monitoring. The Lambda deliberately runs
# outside the VPC -- see the docstring in pathml.monitoring.lambda_handler --
# so this adds no NAT gateway or VPC endpoints, keeping the added cost to
# just the Lambda invocations (a few/day, free-tier), one CloudWatch custom
# metric (~$0.30/mo), two alarms (~$0.20/mo), and SNS (first 1k emails free).

variable "alert_email" {
  type        = string
  description = "Where drift/error-rate alarm notifications go. No default -- pass via TF_VAR_alert_email so it's never committed."
}

variable "drift_check_schedule" {
  type    = string
  default = "rate(6 hours)"
}

variable "drift_share_threshold" {
  type        = number
  default     = 0.5
  description = "Fraction of monitored feature columns that must show as drifted to alarm"
}

variable "api_error_threshold" {
  type        = number
  default     = 5
  description = "Number of 5xx responses in a 5-minute window that triggers the error-rate alarm"
}

# --- ECR repo for the drift-monitor Lambda's container image ---

resource "aws_ecr_repository" "drift_monitor" {
  name                 = "${var.project_name}-drift-monitor"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "drift_monitor" {
  repository = aws_ecr_repository.drift_monitor.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

# --- SNS: where alarms actually notify ---

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --- Lambda execution role ---

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "drift_monitor" {
  name               = "${var.project_name}-drift-monitor"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "drift_monitor_basic_execution" {
  role       = aws_iam_role.drift_monitor.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "drift_monitor" {
  statement {
    sid       = "ReadBaselineAndPredictionImages"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/monitoring/baseline.csv", "${aws_s3_bucket.artifacts.arn}/predictions/*"]
  }

  statement {
    # GetObject alone isn't enough: without ListBucket, S3 can't tell the
    # caller whether a missing key is "doesn't exist" vs "you can't see it",
    # so it returns an opaque AccessDenied instead of NoSuchKey for missing
    # objects -- which breaks the per-image try/except in _load_recent_features.
    # Scoped to just the prefixes this Lambda reads, via the s3:prefix condition.
    sid       = "ListBucketForMissingKeyDetection"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["monitoring/baseline.csv", "predictions/*"]
    }
  }

  statement {
    sid       = "WriteDriftReports"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/monitoring/reports/*"]
  }

  statement {
    sid       = "PublishDriftMetric"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData does not support resource-level scoping
  }

  statement {
    sid       = "ResolveApiTaskIp"
    actions   = ["ecs:ListTasks", "ecs:DescribeTasks"]
    resources = ["*"] # ListTasks/DescribeTasks don't support resource-level scoping either; read-only
  }

  statement {
    sid       = "ResolveApiTaskEni"
    actions   = ["ec2:DescribeNetworkInterfaces"]
    resources = ["*"] # a Describe action -- EC2 doesn't support resource-level ARNs for these
  }
}

resource "aws_iam_role_policy" "drift_monitor" {
  name   = "${var.project_name}-drift-monitor"
  role   = aws_iam_role.drift_monitor.id
  policy = data.aws_iam_policy_document.drift_monitor.json
}

# --- Lambda function ---
# NOTE bootstrap order: aws_lambda_function with package_type=Image requires
# the image tag to already exist in ECR at apply time (unlike ECS, which
# tolerates an empty repo and just fails to start tasks). First apply:
# `terraform apply -target=aws_ecr_repository.drift_monitor`, build+push the
# image, THEN `terraform apply` for everything else. See README.

resource "aws_lambda_function" "drift_monitor" {
  function_name = "${var.project_name}-drift-monitor"
  role          = aws_iam_role.drift_monitor.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.drift_monitor.repository_url}:latest"
  timeout       = 120
  memory_size   = 1024

  environment {
    variables = {
      ECS_CLUSTER          = aws_ecs_cluster.main.name
      ECS_SERVICE          = aws_ecs_service.api.name
      S3_ARTIFACTS_BUCKET  = aws_s3_bucket.artifacts.bucket
      CLOUDWATCH_NAMESPACE = "PathML/Monitoring"
    }
  }

  lifecycle {
    ignore_changes = [image_uri] # cd.yml updates this via `aws lambda update-function-code`, not terraform apply
  }
}

resource "aws_cloudwatch_log_group" "drift_monitor" {
  name              = "/aws/lambda/${aws_lambda_function.drift_monitor.function_name}"
  retention_in_days = 14
}

# --- EventBridge: run the check on a schedule ---

resource "aws_cloudwatch_event_rule" "drift_check_schedule" {
  name                = "${var.project_name}-drift-check"
  schedule_expression = var.drift_check_schedule
}

resource "aws_cloudwatch_event_target" "drift_check" {
  rule = aws_cloudwatch_event_rule.drift_check_schedule.name
  arn  = aws_lambda_function.drift_monitor.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.drift_monitor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.drift_check_schedule.arn
}

# --- CloudWatch alarms: drift + error-rate spike ---

resource "aws_cloudwatch_metric_alarm" "drift" {
  alarm_name          = "${var.project_name}-drift-share-high"
  namespace           = "PathML/Monitoring"
  metric_name         = "DriftShare"
  statistic           = "Maximum"
  period              = 21600 # matches the default 6h schedule -- one data point per run
  evaluation_periods  = 1
  threshold           = var.drift_share_threshold
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching" # a paused/scaled-to-zero service shouldn't page anyone
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# Reuses the API's existing awslogs log group -- no new logging infra, just
# a metric filter turning "500" response lines already being written into a
# CloudWatch metric.
resource "aws_cloudwatch_log_metric_filter" "api_errors" {
  name           = "${var.project_name}-api-errors"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "500"

  metric_transformation {
    name          = "ApiErrorCount"
    namespace     = "PathML/Monitoring"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_errors" {
  alarm_name          = "${var.project_name}-api-error-rate-high"
  namespace           = "PathML/Monitoring"
  metric_name         = "ApiErrorCount"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = var.api_error_threshold
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

output "sns_alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "drift_monitor_ecr_repository_url" {
  value = aws_ecr_repository.drift_monitor.repository_url
}
