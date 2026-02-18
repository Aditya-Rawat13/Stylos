# Project Stylos - Blockchain Module

This module contains the smart contracts and blockchain infrastructure for Project Stylos' proof-of-authorship system.

## Overview

The blockchain module implements Soulbound Tokens (non-transferable NFTs) that serve as immutable proof of academic authorship. Each verified essay submission receives a unique token that cannot be transferred, ensuring permanent attribution to the original author.

## Features

- **Soulbound Tokens**: Non-transferable NFTs for permanent authorship proof
- **Gas Optimization**: Batch operations and optimized storage patterns
- **Upgradeable Contracts**: UUPS proxy pattern for future improvements
- **Role-Based Access Control**: Secure permission system
- **Comprehensive Events**: Detailed logging for monitoring and analytics
- **IPFS Integration**: Decentralized storage for essay content and metadata

## Smart Contract Architecture

### ProofOfAuthorship Contract

The main contract implements:

- **ERC721**: Base NFT functionality (with transfers disabled)
- **AccessControl**: Role-based permissions
- **Pausable**: Emergency pause functionality
- **UUPSUpgradeable**: Upgradeable proxy pattern
- **ReentrancyGuard**: Protection against reentrancy attacks

#### Key Roles

- `DEFAULT_ADMIN_ROLE`: Contract administration
- `VERIFIER_ROLE`: Can mint tokens and verify submissions
- `UPGRADER_ROLE`: Can upgrade contract implementation
- `PAUSER_ROLE`: Can pause/unpause contract

#### Core Functions

- `mintProofToken()`: Mint individual proof-of-authorship token
- `batchMintProofTokens()`: Gas-optimized batch minting
- `updateSubmission()`: Update IPFS hash and scores
- `setVerificationStatus()`: Mark submissions as verified/unverified

## Installation

1. Install dependencies:
```bash
npm install
```

2. Copy environment configuration:
```bash
cp .env.example .env
```

3. Configure your environment variables in `.env`

## Deployment

### Deploy to Polygon Mumbai (Testnet)

```bash
npm run deploy:mumbai
```

### Deploy to Polygon Mainnet

```bash
npm run deploy:polygon
```

### Verify Contract

```bash
npm run verify:polygon
```

## Testing

Run the comprehensive test suite:

```bash
npm test
```

Generate gas report:

```bash
npm run gas-report
```

Run coverage analysis:

```bash
npm run coverage
```

## Contract Management

### Governance Operations

Grant verifier role to an address:
```bash
npx hardhat run scripts/governance.js grant-verifier 0x1234... --network polygon
```

Revoke verifier role:
```bash
npx hardhat run scripts/governance.js revoke-verifier 0x1234... --network polygon
```

Pause contract (emergency):
```bash
npx hardhat run scripts/governance.js emergency-pause --network polygon
```

List role members:
```bash
npx hardhat run scripts/governance.js list-roles verifier --network polygon
```

### Monitoring

Monitor real-time events:
```bash
npx hardhat run scripts/monitor.js monitor --network polygon
```

Get contract metrics:
```bash
npx hardhat run scripts/monitor.js metrics --network polygon
```

Perform health check:
```bash
npx hardhat run scripts/monitor.js health --network polygon
```

Get historical events:
```bash
npx hardhat run scripts/monitor.js history 0 latest --network polygon
```

## Contract Upgrades

The contract uses the UUPS (Universal Upgradeable Proxy Standard) pattern for upgrades:

1. Deploy new implementation
2. Upgrade proxy to point to new implementation
3. Verify upgrade functionality

```bash
npm run upgrade:polygon
```

## Gas Optimization

The contract implements several gas optimization techniques:

- **Packed Structs**: Efficient storage layout
- **Batch Operations**: Reduce per-transaction overhead
- **Custom Errors**: More efficient than require strings
- **Event Optimization**: Indexed parameters for efficient filtering

### Gas Usage Benchmarks

- Single mint: ~150k gas
- Batch mint (5 tokens): ~120k gas per token
- Update submission: ~45k gas
- Role management: ~50k gas

## Security Features

### Access Control

- Role-based permissions with OpenZeppelin's AccessControl
- Multi-signature wallet support for critical operations
- Timelock for sensitive upgrades

### Soulbound Implementation

Tokens are made non-transferable by reverting all transfer functions:
- `transferFrom()`
- `safeTransferFrom()`
- `approve()`
- `setApprovalForAll()`

### Emergency Controls

- Pausable functionality for emergency stops
- Role-based pause/unpause permissions
- Event logging for all administrative actions

## Integration with Backend

The smart contract integrates with the backend through:

1. **Event Monitoring**: Backend listens for blockchain events
2. **Transaction Submission**: Backend submits verification results
3. **Status Updates**: Real-time transaction status tracking
4. **IPFS Coordination**: Synchronized content storage

## Network Configuration

### Polygon Mainnet
- Chain ID: 137
- RPC: https://polygon-rpc.com
- Explorer: https://polygonscan.com

### Polygon Mumbai (Testnet)
- Chain ID: 80001
- RPC: https://rpc-mumbai.maticvigil.com
- Explorer: https://mumbai.polygonscan.com
- Faucet: https://faucet.polygon.technology

## IPFS Integration

Essays and metadata are stored on IPFS with blockchain references:

1. **Content Storage**: Full essay text encrypted and stored on IPFS
2. **Metadata Storage**: Verification results and stylometric data
3. **Hash Linking**: Blockchain stores IPFS hashes for immutable reference
4. **Redundancy**: Multiple IPFS nodes for reliability

## Monitoring and Analytics

### Event Types

- `SubmissionMinted`: New token created
- `BatchSubmissionsMinted`: Multiple tokens created
- `SubmissionVerified`: Verification status updated
- `SubmissionUpdated`: Metadata updated
- `RoleGranted/RoleRevoked`: Permission changes
- `Paused/Unpaused`: Contract state changes

### Metrics Tracked

- Total tokens minted
- Verification success rate
- Gas usage patterns
- Transaction frequency
- Error rates

## Troubleshooting

### Common Issues

1. **Transaction Fails**: Check gas limit and network congestion
2. **Role Errors**: Verify account has required permissions
3. **Upgrade Fails**: Ensure upgrader role and implementation compatibility
4. **Events Missing**: Check block range and network connectivity

### Debug Commands

Check contract status:
```bash
npx hardhat run scripts/governance.js stats --network polygon
```

Verify deployment:
```bash
npx hardhat verify --network polygon <contract-address>
```

## Contributing

1. Follow Solidity style guide
2. Add comprehensive tests for new features
3. Update gas benchmarks
4. Document all public functions
5. Test on testnet before mainnet deployment

## License

MIT License - see LICENSE file for details.