const { run } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Starting contract verification...");

  // Load deployment info
  const deploymentFile = path.join(__dirname, "..", "deployments", `${hre.network.name}.json`);
  
  if (!fs.existsSync(deploymentFile)) {
    throw new Error(`Deployment file not found: ${deploymentFile}`);
  }

  const deploymentInfo = JSON.parse(fs.readFileSync(deploymentFile, "utf8"));
  const contractAddress = deploymentInfo.contractAddress;
  const implementationAddress = deploymentInfo.implementationAddress;

  console.log("Verifying proxy contract at:", contractAddress);
  console.log("Implementation address:", implementationAddress);

  try {
    // Verify the implementation contract
    await run("verify:verify", {
      address: implementationAddress,
      constructorArguments: [],
    });

    console.log("Implementation contract verified successfully!");

    // Note: The proxy contract verification is handled automatically by Hardhat
    // when using the upgrades plugin with etherscan verification

  } catch (error) {
    if (error.message.toLowerCase().includes("already verified")) {
      console.log("Contract is already verified!");
    } else {
      console.error("Verification failed:", error);
      throw error;
    }
  }

  console.log("Verification completed!");
}

// Handle errors
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Verification failed:", error);
    process.exit(1);
  });