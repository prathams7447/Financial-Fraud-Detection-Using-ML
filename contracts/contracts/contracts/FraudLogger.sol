// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

contract FraudLogger {
    // Event emitted when fraud is detected
    event FraudDetected(
        uint256 indexed transactionId,
        uint256 timestamp,
        uint256 fraudScore
    );

    // Struct to store fraud details
    struct FraudRecord {
        uint256 timestamp;
        uint256 fraudScore;
        bool exists;
    }

    // Mapping from transaction ID to fraud record
    mapping(uint256 => FraudRecord) public fraudRecords;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    function logFraud(
        uint256 _transactionId,
        uint256 _fraudScore
    ) external onlyOwner {
        require(_transactionId > 0, "Invalid transaction ID");
        require(_fraudScore <= 100, "Fraud score must be between 0 and 100");

        fraudRecords[_transactionId] = FraudRecord({
            timestamp: block.timestamp,
            fraudScore: _fraudScore,
            exists: true
        });

        emit FraudDetected(_transactionId, block.timestamp, _fraudScore);
    }

    function getFraudRecord(uint256 _transactionId) 
        external 
        view 
        returns (uint256 timestamp, uint256 fraudScore, bool exists) 
    {
        FraudRecord memory record = fraudRecords[_transactionId];
        return (record.timestamp, record.fraudScore, record.exists);
    }
}
