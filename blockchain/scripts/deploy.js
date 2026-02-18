const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Starting deployment of ProofOfAuthorship contract...");

  // Get the deployer account
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);

  // Check balance
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", ethers.formatEther(balance), "ETH");

  // Get the contract factory
  const ProofOfAuthorship = await ethers.getContractFactory("ProofOfAuthorship");

  console.log("Deploying ProofOfAuthorship as upgradeable proxy...");

  // Deploy the upgradeable contract
  const proofOfAuthorship = await upgrades.deployProxy(
    ProofOfAuthorship,
    [deployer.address], // Initialize with deployer as admin
    {
      initializer: "initialize",
      kind: "uups",
    }
  );

  await proofOfAuthorship.waitForDeployment();

  const contractAddress = await proofOfAuthorship.getAddress();
  console.log("ProofOfAuthorship deployed to:", contractAddress);

  // Get implementation address
  const implementationAddress = await upgrades.erc1967.getImplementationAddress(contractAddress);
  console.log("Implementation address:", implementationAddress);

  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    contractAddress: contractAddress,
    implementationAddress: implementationAddress,
    deployer: deployer.address,
    deploymentTime: new Date().toISOString(),
    blockNumber: await ethers.provider.getBlockNumber(),
    transactionHash: proofOfAuthorship.deploymentTransaction()?.hash,
  };

  // Create deployments directory if it doesn't exist
  const deploymentsDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  // Save deployment info to file
  const deploymentFile = path.join(deploymentsDir, `${hre.network.name}.json`);
  fs.writeFileSync(deploymentFile, JSON.stringify(deploymentInfo, null, 2));

  console.log("Deployment info saved to:", deploymentFile);

  // Grant additional roles if needed
  console.log("Setting up roles...");
  
  // You can add additional verifiers here
  // await proofOfAuthorship.grantRole(await proofOfAuthorship.VERIFIER_ROLE(), "0x...");

  console.log("Deployment completed successfully!");

  // If on a live network, wait for confirmations before verification
  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    console.log("Waiting for block confirmations...");
    await proofOfAuthorship.deploymentTransaction()?.wait(5);
    
    console.log("Contract deployed and confirmed. You can now verify it with:");
    console.log(`npx hardhat verify --network ${hre.network.name} ${contractAddress}`);
  }

  return {
    contract: proofOfAuthorship,
    address: contractAddress,
    implementationAddress: implementationAddress,
  };
}

// Handle errors
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Deployment failed:", error);
    process.exit(1);
  });