# Custom domain: ronaldsierra.dev, registered and hosted in this account's
# Route53 already. The API has no ALB (see terraform/ecs.tf), so there's no
# stable address to alias the domain to directly. Instead:
#
#   ronaldsierra.dev (CloudFront, HTTPS) --> origin.ronaldsierra.dev (A record,
#     kept fresh by the dns_updater Lambda below) --> the Fargate task's
#     current public IP, over plain HTTP on var.container_port.
#
# CloudFront is the cheapest way to get a real ACM-backed HTTPS cert in front
# of a target with no fixed IP -- an ALB would be a fixed ~$16+/mo just for
# the thing this already-free distribution does at demo-level traffic.

variable "domain_name" {
  type        = string
  default     = "ronaldsierra.dev"
  description = "Public domain the console/API are served at. Must already be a Route53-hosted zone in this account."
}

variable "origin_subdomain" {
  type        = string
  default     = "origin"
  description = "Internal record the dns_updater Lambda keeps pointed at the Fargate task's current public IP; CloudFront's origin."
}

data "aws_route53_zone" "primary" {
  name         = var.domain_name
  private_zone = false
}

locals {
  origin_fqdn = "${var.origin_subdomain}.${var.domain_name}"
}

# --- ACM certificate (must be requested in us-east-1 for CloudFront) ---

resource "aws_acm_certificate" "site" {
  provider          = aws.us_east_1
  domain_name       = var.domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.site.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      type   = dvo.resource_record_type
      record = dvo.resource_record_value
    }
  }

  zone_id         = data.aws_route53_zone.primary.zone_id
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "site" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.site.arn
  validation_record_fqdns = [for r in aws_route53_record.cert_validation : r.fqdn]
}

# --- origin record: the thing that actually changes, kept fresh by Lambda ---

resource "aws_route53_record" "origin" {
  zone_id = data.aws_route53_zone.primary.zone_id
  name    = local.origin_fqdn
  type    = "A"
  ttl     = 30
  # Placeholder (TEST-NET-3, non-routable) -- overwritten by dns_updater the
  # next time the ECS task reaches RUNNING. Terraform must own record
  # creation (CloudFront needs the name to exist) but must not fight the
  # Lambda's UPSERTs on every subsequent apply.
  records = ["203.0.113.1"]

  lifecycle {
    ignore_changes = [records]
  }
}

# --- CloudFront: terminates HTTPS, proxies to the origin record over HTTP ---

resource "aws_cloudfront_distribution" "site" {
  depends_on      = [aws_route53_record.origin]
  enabled         = true
  is_ipv6_enabled = true
  comment         = "${var.project_name} -- stable HTTPS domain in front of the Fargate task's ephemeral public IP"
  price_class     = "PriceClass_100"
  aliases         = [var.domain_name]

  origin {
    domain_name = local.origin_fqdn
    origin_id   = "ecs-task"

    custom_origin_config {
      http_port                = var.container_port
      https_port               = 443 # required by the provider schema; unused, origin_protocol_policy is http-only
      origin_protocol_policy   = "http-only"
      origin_ssl_protocols     = ["TLSv1.2"]
      origin_read_timeout      = 30
      origin_keepalive_timeout = 5
    }
  }

  default_cache_behavior {
    target_origin_id       = "ecs-task"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]

    # This fronts a live API/console, not static assets -- AWS-managed
    # "CachingDisabled" + "AllViewer" (forward all headers/cookies/query
    # strings) rather than a custom policy neither of which need tuning here.
    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    origin_request_policy_id = "b689b0a8-53d0-40ab-baf2-68738e2966ac"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.site.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
}

resource "aws_route53_record" "apex_alias_a" {
  zone_id = data.aws_route53_zone.primary.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "apex_alias_aaaa" {
  zone_id = data.aws_route53_zone.primary.zone_id
  name    = var.domain_name
  type    = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}

output "site_url" {
  value = "https://${var.domain_name}"
}

# --- dns_updater Lambda: keeps aws_route53_record.origin pointed at the
# --- Fargate task's current public IP, triggered on every task-state-change ---

data "archive_file" "dns_updater" {
  type        = "zip"
  source_file = "${path.module}/lambda_src/dns_updater.py"
  output_path = "${path.module}/lambda_src/dns_updater.zip"
}

resource "aws_iam_role" "dns_updater" {
  name               = "${var.project_name}-dns-updater"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "dns_updater_basic_execution" {
  role       = aws_iam_role.dns_updater.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "dns_updater" {
  statement {
    sid       = "ResolveTaskEni"
    actions   = ["ecs:ListTasks", "ecs:DescribeTasks"]
    resources = ["*"] # ListTasks/DescribeTasks don't support resource-level scoping
  }

  statement {
    sid       = "ResolveEni"
    actions   = ["ec2:DescribeNetworkInterfaces"]
    resources = ["*"] # a Describe action -- EC2 doesn't support resource-level ARNs for these
  }

  statement {
    sid       = "UpdateOriginRecord"
    actions   = ["route53:ChangeResourceRecordSets"]
    resources = ["arn:aws:route53:::hostedzone/${data.aws_route53_zone.primary.zone_id}"]
  }
}

resource "aws_iam_role_policy" "dns_updater" {
  name   = "${var.project_name}-dns-updater"
  role   = aws_iam_role.dns_updater.id
  policy = data.aws_iam_policy_document.dns_updater.json
}

resource "aws_lambda_function" "dns_updater" {
  function_name    = "${var.project_name}-dns-updater"
  role             = aws_iam_role.dns_updater.arn
  runtime          = "python3.12"
  handler          = "dns_updater.handler"
  filename         = data.archive_file.dns_updater.output_path
  source_code_hash = data.archive_file.dns_updater.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      ROUTE53_ZONE_ID    = data.aws_route53_zone.primary.zone_id
      ORIGIN_RECORD_NAME = local.origin_fqdn
      # Literal names, not aws_ecs_cluster.main.name / aws_ecs_service.api.name --
      # see the comment on aws_cloudwatch_event_rule.ecs_task_running for why a
      # resource-attribute reference here is a footgun around aws_db_instance.postgres.
      ECS_CLUSTER = "${var.project_name}-cluster"
      ECS_SERVICE = "${var.project_name}-api"
    }
  }
}

resource "aws_cloudwatch_log_group" "dns_updater" {
  name              = "/aws/lambda/${aws_lambda_function.dns_updater.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_event_rule" "ecs_task_running" {
  name = "${var.project_name}-task-running"
  # Deliberately built from var.project_name/var.aws_region rather than
  # aws_ecs_cluster.main.arn / aws_ecs_service.api.name: a resource-attribute
  # reference here would pull aws_ecs_service.api's full dependency chain
  # (task definition -> SSM param -> aws_db_instance.postgres) into any
  # -target plan that includes this rule, which is a correctness footgun
  # around RDS in particular. Same names either way -- see ecs.tf.
  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      clusterArn = ["arn:aws:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:cluster/${var.project_name}-cluster"]
      lastStatus = ["RUNNING"]
      group      = ["service:${var.project_name}-api"]
    }
  })
}

resource "aws_cloudwatch_event_target" "dns_updater" {
  rule = aws_cloudwatch_event_rule.ecs_task_running.name
  arn  = aws_lambda_function.dns_updater.arn
}

resource "aws_lambda_permission" "allow_eventbridge_dns_updater" {
  statement_id  = "AllowExecutionFromEventBridgeTaskRunning"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dns_updater.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ecs_task_running.arn
}
