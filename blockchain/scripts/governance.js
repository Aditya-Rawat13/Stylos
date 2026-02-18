const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

/**
 * Governance utilities for managing the ProofOfAuthorship contract
 */

async function loadContract() {
  const deploymentFile = path.join(__dirname, "..", "deployments", `${hre.network.name}.json`);
  
  if (!fs.existsSync(deploymentFile)) {
    throw new Error(`Deployment file not found: ${deploymentFile}`);
  }

  const deploymentInfo = JSON.parse(fs.readFileSync(deploymentFile, "utf8"));
  const contract = await ethers.getContractAt("ProofOfAuthorship", deploymentInfo.contractAddress);
  
  return { contract, deploymentInfo };
}

async function grantVerifierRole(verifierAddress) {
  console.log(`Granting VERIFIER_ROLE to ${verifierAddress}...`);
  
  const { contract } = await loadContract();
  const [admin] = await ethers.getSigners();
  
  const VERIFIER_ROLE = await contract.VERIFIER_ROLE();
  
  // Check if already has role
  if (await contract.hasRole(VERIFIER_ROLE, verifierAddress)) {
    console.log("Address already has VERIFIER_ROLE");
    return;
  }
  
  const tx = await contract.connect(admin).grantRole(VERIFIER_ROLE, verifierAddress);
  await tx.wait();
  
  console.log(`VERIFIER_ROLE granted to ${verifierAddress}`);
  console.log(`Transaction hash: ${tx.hash}`);
}

async function revokeVerifierRole(verifierAddress) {
  console.log(`Revoking VERIFIER_ROLE from ${verifierAddress}...`);
  
  const { contract } = await loadContract();
  const [admin] = await ethers.getSigners();
  
  const VERIFIER_ROLE = await contract.VERIFIER_ROLE();
  
  const tx = await contract.connect(admin).revokeRole(VERIFIER_ROLE, verifierAddress);
  await tx.wait();
  
  console.log(`VERIFIER_ROLE revoked from ${verifierAddress}`);
  console.log(`Transaction hash: ${tx.hash}`);
}

async function listRoleMembers(roleName) {
  const { contract } = await loadContract();
  
  let roleHash;
  switch (roleName.toUpperCase()) {
    case "ADMIN":
      roleHash = await contract.DEFAULT_ADMIN_ROLE();
      break;
    case "VERIFIER":
      roleHash = await contract.VERIFIER_ROLE();
      break;
    case "UPGRADER":
      roleHash = await contract.UPGRADER_ROLE();
      break;
    case "PAUSER":
      roleHash = await contract.PAUSER_ROLE();
      break;
    default:
      throw new Error(`Unknown role: ${roleName}`);
  }
  
  console.log(`Members of ${roleName} role:`);
  
  // Note: This is a simplified approach. In production, you'd want to use events
  // or maintain an off-chain registry for better performance
  const roleMembers = [];
  
  // Check some common addresses (in production, use event logs)
  const [deployer] = await ethers.getSigners();
  if (await contract.hasRole(roleHash, deployer.address)) {
    roleMembers.push(deployer.address);
  }
  
  console.log(roleMembers);
  return roleMembers;
}

async function pauseContract() {
  console.log("Pausing contract...");
  
  const { contract } = await loadContract();
  const [admin] = await ethers.getSigners();
  
  if (await contract.paused()) {
    console.log("Contract is already paused");
    return;
  }
  
  const tx = await contract.connect(admin).pause();
  await tx.wait();
  
  console.log("Contract paused successfully");
  console.log(`Transaction hash: ${tx.hash}`);
}

async function unpauseContract() {
  console.log("Unpausing contract...");
  
  const { contract } = await loadContract();
  const [admin] = await ethers.getSigners();
  
  if (!(await contract.paused())) {
    console.log("Contract is not paused");
    return;
  }
  
  const tx = await contract.connect(admin).unpause();
  await tx.wait();
  
  console.log("Contract unpaused successfully");
  console.log(`Transaction hash: ${tx.hash}`);
}

async function getContractStats() {
  const { contract } = await loadContract();
  
  const totalSupply = await contract.totalSupply();
  const isPaused = await contract.paused();
  
  console.log("Contract Statistics:");
  console.log(`Total tokens minted: ${totalSupply}`);
  console.log(`Contract paused: ${isPaused}`);
  console.log(`Contract address: ${await contract.getAddress()}`);
  
  return {
    totalSupply: totalSupply.toString(),
    isPaused,
    contractAddress: await contract.getAddress(),
  };
}

async function emergencyPause() {
  console.log("EMERGENCY PAUSE - Pausing contract immediately...");
  
  const { contract } = await loadContract();
  const [admin] = await ethers.getSigners();
  
  // Use higher gas price for emergency
  const tx = await contract.connect(admin).pause({
    gasPrice: ethers.parseUnits("50", "gwei"),
  });
  
  console.log(`Emergency pause transaction sent: ${tx.hash}`);
  console.log("Waiting for confirmation...");
  
  await tx.wait();
  console.log("Emergency pause confirmed!");
}

// CLI interface
async function main() {
  const command = process.argv[2];
  const arg = process.argv[3];
  
  try {
    switch (command) {
      case "grant-verifier":
        if (!arg) throw new Error("Please provide verifier address");
        await grantVerifierRole(arg);
        break;
        
      case "revoke-verifier":
        if (!arg) throw new Error("Please provide verifier address");
        await revokeVerifierRole(arg);
        break;
        
      case "list-roles":
        if (!arg) throw new Error("Please provide role name (admin, verifier, upgrader, pauser)");
        await listRoleMembers(arg);
        break;
        
      case "pause":
        await pauseContract();
        break;
        
      case "unpause":
        await unpauseContract();
        break;
        
      case "stats":
        await getContractStats();
        break;
        
      case "emergency-pause":
        await emergencyPause();
        break;
        
      default:
        console.log("Available commands:");
        console.log("  grant-verifier <address>   - Grant VERIFIER_ROLE to address");
        console.log("  revoke-verifier <address>  - Revoke VERIFIER_ROLE from address");
        console.log("  list-roles <role>          - List members of role");
        console.log("  pause                      - Pause contract");
        console.log("  unpause                    - Unpause contract");
        console.log("  stats                      - Show contract statistics");
        console.log("  emergency-pause            - Emergency pause with high gas");
    }
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

// Export functions for use in other scripts
module.exports = {
  grantVerifierRole,
  revokeVerifierRole,
  listRoleMembers,
  pauseContract,
  unpauseContract,
  getContractStats,
  emergencyPause,
  loadContract,
};

// Run CLI if called directly
if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}