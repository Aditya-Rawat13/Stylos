const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Starting contract upgrade...");

  // Get the deployer account
  const [deployer] = await ethers.getSigners();
  console.log("Upgrading contract with account:", deployer.address);

  // Load existing deployment info
  const deploymentFile = path.join(__dirname, "..", "deployments", `${hre.network.name}.json`);
  
  if (!fs.existsSync(deploymentFile)) {
    throw new Error(`Deployment file not found: ${deploymentFile}`);
  }

  const deploymentInfo = JSON.parse(fs.readFileSync(deploymentFile, "utf8"));
  const proxyAddress = deploymentInfo.contractAddress;

  console.log("Existing proxy address:", proxyAddress);

  // Get the new contract factory
  const ProofOfAuthorshipV2 = await ethers.getContractFactory("ProofOfAuthorship");

  console.log("Upgrading ProofOfAuthorship...");

  // Upgrade the contract
  const upgraded = await upgrades.upgradeProxy(proxyAddress, ProofOfAuthorshipV2);
  await upgraded.waitForDeployment();

  console.log("Contract upgraded successfully!");

  // Get new implementation address
  const newImplementationAddress = await upgrades.erc1967.getImplementationAddress(proxyAddress);
  console.log("New implementation address:", newImplementationAddress);

  // Update deployment info
  const upgradeInfo = {
    ...deploymentInfo,
    previousImplementationAddress: deploymentInfo.implementationAddress,
    implementationAddress: newImplementationAddress,
    upgradeTime: new Date().toISOString(),
    upgradeBlockNumber: await ethers.provider.getBlockNumber(),
    upgrader: deployer.address,
  };

  // Save updated deployment info
  fs.writeFileSync(deploymentFile, JSON.stringify(upgradeInfo, null, 2));

  console.log("Upgrade info saved to:", deploymentFile);

  // Verify the upgrade worked
  const contract = await ethers.getContractAt("ProofOfAuthorship", proxyAddress);
  const totalSupply = await contract.totalSupply();
  console.log("Contract is functional. Total supply:", totalSupply.toString());

  console.log("Upgrade completed successfully!");

  return {
    contract: upgraded,
    address: proxyAddress,
    newImplementationAddress: newImplementationAddress,
  };
}

// Handle errors
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Upgrade failed:", error);
    process.exit(1);
  });