resource "random_password" "db" {
  length  = 24
  special = false # keep the DSN URL-parseable without percent-encoding
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnets"
  subnet_ids = data.aws_subnets.default.ids
}

resource "aws_db_instance" "postgres" {
  identifier     = "${var.project_name}-db"
  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage = var.db_allocated_storage
  storage_type      = "gp3"

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Demo/portfolio project, not a durable production database: skip the
  # final snapshot and keep no backups so `terraform destroy` fully tears
  # this down with no leftover (billed) storage.
  skip_final_snapshot     = true
  backup_retention_period = 0
  apply_immediately       = true
}

# The ECS task execution role reads this at container start and injects it
# as the DATABASE_URL env var -- see the `secrets` block in ecs.tf. Kept out
# of the task definition / state as plaintext by using SecureString.
resource "aws_ssm_parameter" "database_url" {
  name  = "/${var.project_name}/database_url"
  type  = "SecureString"
  value = "postgresql://${var.db_username}:${random_password.db.result}@${aws_db_instance.postgres.address}:5432/${var.db_name}"
}
