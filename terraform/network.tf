# Reuse the account's default VPC rather than provisioning a new one --
# its subnets are already public (route to an internet gateway, auto-assign
# public IP), which lets the Fargate task pull images and be reached directly
# with no ALB or NAT gateway, both of which have fixed monthly costs this
# small a project doesn't need.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_caller_identity" "current" {}
