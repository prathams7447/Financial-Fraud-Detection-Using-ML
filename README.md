# Financial Fraud Detection Using Machine Learning, Blockchain, and Kafka

This project implements a real-time financial fraud detection system using machine learning models, Apache Kafka for stream processing, and blockchain for immutable fraud logging.

## Project Structure
```
fraud_detection/
├── data/                      # Store IEEE-CIS dataset here
├── models/                    # Trained ML models
├── contracts/                 # Solidity smart contracts
│   └── FraudLogger.sol       # Fraud logging contract
├── src/
│   ├── preprocessing.py       # Data preprocessing and feature engineering
│   ├── train_model.py        # Model training script
│   ├── kafka_producer.py     # Kafka transaction producer
│   ├── kafka_consumer.py     # Fraud detection consumer
│   ├── blockchain_logger.py  # Blockchain logging functionality
│   ├── app.py               # Flask dashboard application
│   ├── templates/           # HTML templates
│   │   └── index.html      # Dashboard template
│   └── utils.py             # Utility functions
└── requirements.txt          # Project dependencies
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the IEEE-CIS dataset and place in the `data/` directory:
   - train_transaction.csv
   - train_identity.csv

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update Twilio configuration for SMS alerts
   - Configure blockchain settings:
     - Set Web3 provider URI (e.g., Ganache or Sepolia)
     - Add Ethereum private key
     - Set deployed contract address

4. Set up blockchain:
   - Install and start Ganache for local development
   - Deploy FraudLogger.sol contract
   - Update CONTRACT_ADDRESS in .env

5. Start Kafka server (requires Apache Kafka installation)

6. Run the preprocessing and model training:
```bash
python src/preprocessing.py
python src/train_model.py
```

7. Start the Kafka producer and consumer:
```bash
python src/kafka_producer.py
python src/kafka_consumer.py
```

8. Start the dashboard:
```bash
python src/app.py
```
Visit http://localhost:5000 to view the dashboard

## Features
- Real-time transaction simulation using Kafka
- XGBoost + Isolation Forest ensemble for fraud detection
- SHAP-based model explainability
- Blockchain logging of fraud detections
- Real-time SMS alerts via Twilio
- Interactive dashboard with live transaction monitoring
- Feature importance visualization
- Ensemble ML model (XGBoost + Isolation Forest)
- Fraud detection and alerting system
- Transaction logging for flagged transactions
- Email notifications for detected fraud
