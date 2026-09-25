import json
import boto3
import base64
import os
import uuid

s3 = boto3.client('s3')
textract = boto3.client('textract')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

BUCKET_NAME = os.environ.get('BUCKET_NAME', 'dengue-rag-assistant-bucket')
KNOWLEDGE_BASE_ID = os.environ.get('KNOWLEDGE_BASE_ID', '')
MODEL_ARN = os.environ.get('MODEL_ARN', 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0')
GUARDRAIL_ID = os.environ.get('GUARDRAIL_ID', '')
GUARDRAIL_VERSION = os.environ.get('GUARDRAIL_VERSION', '1')

def lambda_handler(event, context):
    path = event.get('path', '')
    http_method = event.get('httpMethod', '')
    
    # Handle CORS Preflight
    if http_method == 'OPTIONS':
        return build_response(200, "OK")
    
    try:
        if path == '/upload' and http_method == 'POST':
            return handle_upload(event)
        elif path == '/ask' and http_method == 'POST':
            return handle_ask(event)
        else:
            return build_response(404, "Not Found")
    except Exception as e:
        print(f"Error: {str(e)}")
        return build_response(500, f"Internal Server Error: {str(e)}")

def handle_upload(event):
    body = json.loads(event.get('body', '{}'))
    file_content = body.get('file', '')
    filename = body.get('filename', f"{uuid.uuid4()}.pdf")
    
    file_bytes = base64.b64decode(file_content)
    file_extension = filename.split('.')[-1].lower()
    
    # Textract Integration for Images
    if file_extension in ['jpg', 'jpeg', 'png']:
        response = textract.detect_document_text(Document={'Bytes': file_bytes})
        extracted_text = "\n".join([item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])
        
        # Save extracted text to reports/
        txt_filename = f"{filename.split('.')[0]}_extracted.txt"
        s3.put_object(Bucket=BUCKET_NAME, Key=f"reports/{txt_filename}", Body=extracted_text.encode('utf-8'))
        return build_response(200, {"message": "Image processed with Textract and text saved.", "filename": txt_filename})
    else:
        # Save directly to reports/ for Bedrock KB sync
        s3.put_object(Bucket=BUCKET_NAME, Key=f"reports/{filename}", Body=file_bytes)
        return build_response(200, {"message": "PDF uploaded successfully.", "filename": filename})

def handle_ask(event):
    body = json.loads(event.get('body', '{}'))
    question = body.get('question', '')
    session_id = body.get('session_id')
    
    # Bedrock Knowledge Base Retrieval and Generation
    rag_config = {
        'knowledgeBaseId': KNOWLEDGE_BASE_ID,
        'modelArn': MODEL_ARN,
        'retrievalConfiguration': {
            'vectorSearchConfiguration': {'numberOfResults': 5}
        }
    }
    
    # Add Guardrails
    if GUARDRAIL_ID:
        rag_config['guardrailConfiguration'] = {
            'guardrailId': GUARDRAIL_ID,
            'guardrailVersion': GUARDRAIL_VERSION
        }
    
    request_params = {
        'input': {'text': question},
        'retrieveAndGenerateConfiguration': {
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': rag_config
        }
    }
    
    if session_id:
        request_params['sessionId'] = session_id
        
    response = bedrock_agent_runtime.retrieve_and_generate(**request_params)
    
    answer = response['output']['text']
    citations = response.get('citations', [])
    new_session_id = response.get('sessionId', '')
    
    # Extract evidence chunks for frontend
    evidence = []
    for citation in citations:
        for ref in citation.get('retrievedReferences', []):
            evidence.append({
                'content': ref['content']['text'],
                'source': ref['location']['s3Location']['uri']
            })
            
    return build_response(200, {
        "answer": answer,
        "evidence": evidence,
        "session_id": new_session_id
    })

def build_response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        'body': json.dumps(body)
    }
