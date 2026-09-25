# AWS Bedrock Multimodal Dengue Report RAG Assistant

This project is a fully-featured, production-ready implementation of a healthcare RAG assistant on AWS, moving away from local FAISS/Ollama setups.

## Key Technologies
- **Vector Database**: Amazon OpenSearch Serverless
- **Embedding Model**: Amazon Titan Embeddings V2
- **Generative AI Model**: Anthropic Claude 3 Haiku (via Amazon Bedrock)
- **Safety**: Amazon Bedrock Guardrails (Healthcare Guardrail)
- **OCR**: Amazon Textract
- **Compute**: AWS Lambda & API Gateway
- **Frontend**: React SPA

## Documentation
Please refer to the `aws_architecture_and_deployment.md` file in the root for a comprehensive deployment guide, architecture diagram, IAM setup, and cost estimation.

## Getting Started

### 1. Generate the Synthetic Dataset
```bash
cd dataset_generator
pip install -r requirements.txt
python generate_data.py
```
This generates 50 sample PDF reports and a CSV file inside `dataset_generator/output/`.

### 2. Deploy Infrastructure
Deploy the CloudFormation template or run Terraform:
```bash
cd infrastructure/terraform
terraform init
terraform apply
```

### 3. Deploy Backend
Zip the `backend/` folder contents and deploy to your AWS Lambda function.

### 4. Run Frontend
```bash
cd frontend
npm install
npm start
```
*Note: Ensure you update `API_URL` in `src/components/Dashboard.js` to your actual API Gateway endpoint.*
