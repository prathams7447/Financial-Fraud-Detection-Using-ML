import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SMSNotifier:
    def __init__(self):
        """Initialize Twilio configuration from environment variables"""
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.to_number = os.getenv('TWILIO_TO_NUMBER')
        self.client = None
        
        if all([self.account_sid, self.auth_token]):
            self.client = Client(self.account_sid, self.auth_token)
        
    def send_fraud_alert(self, transaction_id, amount, confidence):
        """Send fraud alert SMS"""
        if not all([self.client, self.from_number, self.to_number]):
            print("Twilio configuration missing. Skipping SMS notification.")
            return False
            
        try:
            # Create message body
            message_body = (
                f"🚨 FRAUD ALERT!\n\n"
                f"Transaction ID: {transaction_id}\n"
                f"Amount: ${amount:.2f}\n"
                f"Fraud Confidence: {confidence:.2%}\n\n"
                f"Please review immediately."
            )
            
            # Send SMS
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=self.to_number
            )
                
            print(f"Fraud alert SMS sent for Transaction {transaction_id} (SID: {message.sid})")
            return True
            
        except Exception as e:
            print(f"Failed to send SMS alert: {str(e)}")
            return False
