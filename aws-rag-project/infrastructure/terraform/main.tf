provider "aws" {
  region = var.aws_region
}

# S3 Bucket
resource "aws_s3_bucket" "dengue_bucket" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.dengue_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "block_public" {
  bucket = aws_s3_bucket.dengue_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sse" {
  bucket = aws_s3_bucket.dengue_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Create Folders (S3 Objects)
resource "aws_s3_object" "reports_folder" {
  bucket = aws_s3_bucket.dengue_bucket.id
  key    = "reports/"
}

resource "aws_s3_object" "uploads_folder" {
  bucket = aws_s3_bucket.dengue_bucket.id
  key    = "uploads/"
}

resource "aws_s3_object" "processed_folder" {
  bucket = aws_s3_bucket.dengue_bucket.id
  key    = "processed/"
}

resource "aws_s3_object" "logs_folder" {
  bucket = aws_s3_bucket.dengue_bucket.id
  key    = "logs/"
}

# API Gateway
resource "aws_api_gateway_rest_api" "dengue_api" {
  name        = "DengueAssistantAPI"
  description = "API for Dengue RAG Assistant"
}

# Add API Resources and Methods (simplified for demonstration)
