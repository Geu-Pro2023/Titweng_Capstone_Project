import smtplib
import os
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

load_dotenv()

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Titweng Cattle System")

def send_registration_email(to_email: str, owner_name: str, cow_tag: str, pdf_receipt: bytes):
    """Send registration confirmation email with PDF receipt attachment"""
    
    if not all([SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL]):
        print("Email configuration missing - skipping email")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg['To'] = to_email
        msg['Subject'] = f"Cattle Registration Confirmation - {cow_tag}"
        
        # Email body
        body = f"""
Dear {owner_name},

Your cattle registration has been completed successfully!

Registration Details:
- Cow Tag: {cow_tag}
- Owner: {owner_name}
- Status: REGISTERED
- Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}

Please find your registration receipt attached to this email.
Keep this receipt as proof of registration.

Thank you for using Titweng Cattle Recognition System.

Best regards,
Titweng Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF receipt
        pdf_attachment = MIMEApplication(pdf_receipt, _subtype='pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=f'{cow_tag}_receipt.pdf')
        msg.attach(pdf_attachment)
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"Registration email sent to {to_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False