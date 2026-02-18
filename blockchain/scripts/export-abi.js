const fs = require('fs');
const path = require('path');

async function main() {
  console.log('Exporting contract ABI...');

  // Read the compiled contract artifact
  const artifactPath = path.join(
    __dirname,
    '..',
    'artifacts',
    'contracts',
    'ProofOfAuthorship.sol',
    'ProofOfAuthorship.json'
  );

  if (!fs.existsSync(artifactPath)) {
    console.error('Contract artifact not found. Please compile the contract first.');
    console.error('Run: npx hardhat compile');
    process.exit(1);
  }

  const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));

  // Read deployment info
  const deploymentPath = path.join(__dirname, '..', 'deployments', 'localhost.json');
  let contractAddress = '';

  if (fs.existsSync(deploymentPath)) {
    const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
    contractAddress = deployment.contractAddress;
  }

  // Create ABI export
  const abiExport = {
    contractName: 'ProofOfAuthorship',
    abi: artifact.abi,
    bytecode: artifact.bytecode,
    address: contractAddress,
    network: 'localhost',
    chainId: 31337,
  };

  // Save to frontend public directory
  const frontendAbiPath = path.join(__dirname, '..', '..', 'frontend', 'public', 'contracts');
  if (!fs.existsSync(frontendAbiPath)) {
    fs.mkdirSync(frontendAbiPath, { recursive: true });
  }

  fs.writeFileSync(
    path.join(frontendAbiPath, 'ProofOfAuthorship.json'),
    JSON.stringify(abiExport, null, 2)
  );

  // Save to backend config directory
  const backendAbiPath = path.join(__dirname, '..', '..', 'backend', 'config', 'contracts');
  if (!fs.existsSync(backendAbiPath)) {
    fs.mkdirSync(backendAbiPath, { recursive: true });
  }

  fs.writeFileSync(
    path.join(backendAbiPath, 'ProofOfAuthorship.json'),
    JSON.stringify(abiExport, null, 2)
  );

  console.log('✅ ABI exported successfully!');
  console.log('Frontend:', path.join(frontendAbiPath, 'ProofOfAuthorship.json'));
  console.log('Backend:', path.join(backendAbiPath, 'ProofOfAuthorship.json'));
  console.log('Contract Address:', contractAddress);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
