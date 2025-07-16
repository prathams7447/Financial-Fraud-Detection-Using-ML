# Detailed Setup Guide

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Set Up Twilio
1. Sign up for a Twilio account at https://www.twilio.com
2. Get your Account SID and Auth Token from the Twilio Console
3. Buy a phone number or use a trial number
4. Copy `.env.example` to `.env` and update Twilio settings:
```
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM_NUMBER=your_twilio_phone_number
TWILIO_TO_NUMBER=recipient_phone_number
```

## 3. Set Up Blockchain (Local Development with Ganache)

### Install Ganache
1. Download and install Ganache from https://trufflesuite.com/ganache/
2. Launch Ganache and create a new workspace
3. Keep note of the RPC Server URL (default: http://127.0.0.1:8545)

### Deploy Smart Contract
1. Install Node.js and npm if not already installed
2. Install Truffle globally:
```bash
npm install -g truffle
```

3. Create a new Truffle project in the contracts directory:
```bash
cd contracts
truffle init
```

4. Copy FraudLogger.sol to contracts/ directory
5. Create migration file in migrations/2_deploy_contracts.js:
```javascript
const FraudLogger = artifacts.require("FraudLogger");

module.exports = function(deployer) {
  deployer.deploy(FraudLogger);
};
```

6. Deploy the contract:
```bash
truffle migrate --network development
```

7. Copy the deployed contract address from the migration output

### Update Blockchain Configuration
Update your `.env` file with blockchain settings:
```
WEB3_PROVIDER_URI=http://127.0.0.1:8545
ETHEREUM_PRIVATE_KEY=your_private_key_from_ganache
CONTRACT_ADDRESS=your_deployed_contract_address
```

To get the private key:
1. Open Ganache
2. Click on the key icon next to any account
3. Copy the private key

## 4. Set Up Apache Kafka 4.0.0 (KRaft Mode)

### Windows Installation
1. Download Kafka 4.0.0 binary:
   - Go to https://kafka.apache.org/downloads
   - Download "Scala 2.13 - kafka_2.13-4.0.0.tgz"
   - Or use direct link: https://downloads.apache.org/kafka/4.0.0/kafka_2.13-4.0.0.tgz

2. Extract to C:\kafka

3. Create data directory:
```bash
mkdir C:\kafka\data\kafka
```

4. Update config/server.properties:
```properties
# Update the log directory
log.dirs=C:/kafka/data/kafka

# KRaft mode settings
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9093
```

5. Generate a Cluster ID:
```bash
cd C:\kafka
bin\windows\kafka-storage.bat random-uuid
```

6. Format the storage directory with the generated UUID:
```bash
bin\windows\kafka-storage.bat format -t <CLUSTER_ID> -c config\server.properties
```

### Start Kafka Service
1. Start the Kafka Server:
```bash
cd C:\kafka
bin\windows\kafka-server-start.bat config\server.properties
```

2. Create the transactions topic (in another terminal):
```bash
cd C:\kafka
bin\windows\kafka-topics.bat --create --topic transactions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

3. Verify the topic was created:
```bash
bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092
```

## 5. Running the System

1. Start all required services:
   - Ganache should be running
   - Zookeeper and Kafka servers should be running

2. Run the preprocessing and training:
```bash
python src/preprocessing.py
python src/train_model.py
```

3. Start the components in separate terminals:
```bash
# Terminal 1: Kafka Producer
python src/kafka_producer.py

# Terminal 2: Kafka Consumer
python src/kafka_consumer.py

# Terminal 3: Dashboard
python src/app.py
```

4. Access the dashboard at http://localhost:5000

## Troubleshooting

### Kafka Issues
- Ensure Zookeeper is running before starting Kafka
- Check if the topic exists:
```bash
bin\windows\kafka-topics.bat --list --bootstrap-server localhost:9092
```
- Verify port 9092 is not in use by another process

### Blockchain Issues
- Ensure Ganache is running and accessible
- Verify the contract deployment was successful
- Check if the private key has sufficient funds in Ganache
- Make sure the contract address in .env matches the deployed contract

### Twilio Issues
- Verify your Twilio account is active
- Ensure the phone numbers are in the correct format (+1234567890)
- Check if you have sufficient Twilio credits
