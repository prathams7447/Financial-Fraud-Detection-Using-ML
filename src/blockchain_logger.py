from web3 import Web3
from eth_account import Account
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BlockchainLogger:
    def __init__(self):
        """Initialize Web3 and smart contract"""
        # Connect to local Ganache or other network
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER_URI', 'http://127.0.0.1:8545')))
        
        # Set up account
        private_key = os.getenv('ETHEREUM_PRIVATE_KEY')
        self.account = Account.from_key(private_key)
        
        # Load contract ABI and address
        with open('../contracts/build/contracts/FraudLogger.json') as f:
            contract_json = json.load(f)
        contract_address = os.getenv('CONTRACT_ADDRESS')
        
        # Initialize contract
        self.contract = self.w3.eth.contract(
            address=contract_address,
            abi=contract_json['abi']
        )
        
    def log_fraud(self, transaction_id, timestamp, fraud_score):
        """Log fraud detection to blockchain"""
        try:
            # Prepare transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Build transaction
            transaction = self.contract.functions.logFraudDetection(
                transaction_id,
                timestamp,
                int(fraud_score * 100)  # Convert to percentage
            ).build_transaction({
                'chainId': 1337,  # Ganache default
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, private_key=self.account.key
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"Fraud logged to blockchain. Transaction hash: {receipt.transactionHash.hex()}")
            return receipt.transactionHash.hex()
            
        except Exception as e:
            print(f"Failed to log fraud to blockchain: {str(e)}")
            return None
            
    def get_fraud_logs(self, from_block=0):
        """Get all fraud logs from the contract"""
        try:
            fraud_filter = self.contract.events.FraudDetected.create_filter(fromBlock=from_block)
            events = fraud_filter.get_all_entries()
            
            logs = []
            for event in events:
                logs.append({
                    'transaction_id': event.args.transactionId,
                    'timestamp': event.args.timestamp,
                    'fraud_score': event.args.fraudScore / 100,  # Convert back to decimal
                    'block_number': event.blockNumber,
                    'tx_hash': event.transactionHash.hex()
                })
                
            return logs
            
        except Exception as e:
            print(f"Failed to get fraud logs: {str(e)}")
            return []
