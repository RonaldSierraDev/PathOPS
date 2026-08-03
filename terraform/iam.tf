data "aws_iam_policy_document" "ecs_tasks_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role: what ECS itself uses to pull the image and start the
# container (not the app's own permissions -- that's the task role, below).
resource "aws_iam_role" "ecs_task_execution" {
  name               = "${var.project_name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "ecs_task_execution_ssm" {
  statement {
    actions   = ["ssm:GetParameters"]
    resources = [aws_ssm_parameter.database_url.arn]
  }
}

resource "aws_iam_role_policy" "ecs_task_execution_ssm" {
  name   = "${var.project_name}-ecs-execution-ssm"
  role   = aws_iam_role.ecs_task_execution.id
  policy = data.aws_iam_policy_document.ecs_task_execution_ssm.json
}

# Task role: what the running application (the FastAPI process) is allowed
# to do -- download the model checkpoint, and write back the images behind
# any feedback correction (see PREDICTION_IMAGES_S3_BUCKET in pathml.api.main).
resource "aws_iam_role" "ecs_task" {
  name               = "${var.project_name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

data "aws_iam_policy_document" "ecs_task_s3" {
  statement {
    sid       = "ReadModelArtifacts"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/models/*"]
  }

  statement {
    sid       = "WritePredictionImages"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/predictions/*"]
  }

  # The console's Drift view reads back what the monitor Lambda wrote: the
  # report listing, plus GetObject so the API can hand out presigned report
  # URLs (a presigned URL only works if the signing role could fetch the
  # object itself). Read-only, and scoped to the reports prefix -- the API
  # has no reason to see baseline.csv or anything else under monitoring/.
  statement {
    sid       = "ReadDriftReports"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/${local.drift_reports_prefix}*"]
  }

  statement {
    sid       = "ListDriftReports"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${local.drift_reports_prefix}*"]
    }
  }
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name   = "${var.project_name}-ecs-task-s3"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.ecs_task_s3.json
}

# GetMetricStatistics supports neither resource-level permissions nor a
# namespace condition: cloudwatch:namespace is only populated for publishing
# actions (PutMetricData), so gating a read on it silently denies every call
# -- an absent condition key makes StringEquals false, which is an implicit
# deny, not a pass. There is no way to scope this action, so it's granted
# outright; it's read-only over metric data this project publishes anyway.
data "aws_iam_policy_document" "ecs_task_cloudwatch" {
  statement {
    sid       = "ReadDriftMetric"
    actions   = ["cloudwatch:GetMetricStatistics"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ecs_task_cloudwatch" {
  name   = "${var.project_name}-ecs-task-cloudwatch"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.ecs_task_cloudwatch.json
}
