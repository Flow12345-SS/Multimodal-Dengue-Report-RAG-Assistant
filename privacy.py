import re

def mask_sensitive_data(text: str) -> str:
    """
    Masks sensitive information such as phone numbers, emails, PAN, and Aadhaar numbers.
    Replaces detected patterns with [REDACTED].
    """
    if not isinstance(text, str):
        return text
        
    # Mask Email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
    
    # Mask Indian Phone Numbers (e.g., +91 9876543210, 9876543210)
    phone_pattern = r'(?:\+?91[\-\s]?)?[6-9]\d{9}'
    text = re.sub(phone_pattern, '[REDACTED_PHONE]', text)
    
    # Mask PAN Card Number (Indian format: 5 letters, 4 digits, 1 letter)
    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
    text = re.sub(pan_pattern, '[REDACTED_PAN]', text)
    
    # Mask Aadhaar Number (Indian format: 12 digits, often formatted as 4-4-4)
    aadhaar_pattern = r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
    text = re.sub(aadhaar_pattern, '[REDACTED_AADHAAR]', text)
    
    return text
