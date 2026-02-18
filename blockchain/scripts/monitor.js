const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

/**
 * Monitoring and event logging utilities for ProofOfAuthorship contract
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

async function monitorEvents(fromBlock = "latest") {
  console.log("Starting event monitoring...");
  
  const { contract } = await loadContract();
  
  // Listen for SubmissionMinted events
  contract.on("SubmissionMinted", (tokenId, student, contentHash, ipfsHash, authorshipScore, aiProbability, event) => {
    console.log("\n🎯 New Submission Minted:");
    console.log(`  Token ID: ${tokenId}`);
    console.log(`  Student: ${student}`);
    console.log(`  Content Hash: ${contentHash}`);
    console.log(`  IPFS Hash: ${ipfsHash}`);
    console.log(`  Authorship Score: ${authorshipScore}%`);
    console.log(`  AI Probability: ${aiProbability}%`);
    console.log(`  Block: ${event.blockNumber}`);
    console.log(`  Transaction: ${event.transactionHash}`);
    
    // Log to file
    logEvent("SubmissionMinted", {
      tokenId: tokenId.toString(),
      student,
      contentHash,
      ipfsHash,
      authorshipScore,
      aiProbability,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  // Listen for BatchSubmissionsMinted events
  contract.on("BatchSubmissionsMinted", (tokenIds, students, timestamp, event) => {
    console.log("\n📦 Batch Submissions Minted:");
    console.log(`  Token IDs: ${tokenIds.join(", ")}`);
    console.log(`  Students: ${students.length}`);
    console.log(`  Timestamp: ${new Date(Number(timestamp) * 1000).toISOString()}`);
    console.log(`  Block: ${event.blockNumber}`);
    console.log(`  Transaction: ${event.transactionHash}`);
    
    logEvent("BatchSubmissionsMinted", {
      tokenIds: tokenIds.map(id => id.toString()),
      students,
      timestamp: timestamp.toString(),
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      logTimestamp: new Date().toISOString(),
    });
  });

  // Listen for SubmissionVerified events
  contract.on("SubmissionVerified", (tokenId, verifier, verified, event) => {
    console.log("\n✅ Submission Verification Updated:");
    console.log(`  Token ID: ${tokenId}`);
    console.log(`  Verifier: ${verifier}`);
    console.log(`  Verified: ${verified}`);
    console.log(`  Block: ${event.blockNumber}`);
    console.log(`  Transaction: ${event.transactionHash}`);
    
    logEvent("SubmissionVerified", {
      tokenId: tokenId.toString(),
      verifier,
      verified,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  // Listen for SubmissionUpdated events
  contract.on("SubmissionUpdated", (tokenId, newIpfsHash, newAuthorshipScore, event) => {
    console.log("\n🔄 Submission Updated:");
    console.log(`  Token ID: ${tokenId}`);
    console.log(`  New IPFS Hash: ${newIpfsHash}`);
    console.log(`  New Authorship Score: ${newAuthorshipScore}%`);
    console.log(`  Block: ${event.blockNumber}`);
    console.log(`  Transaction: ${event.transactionHash}`);
    
    logEvent("SubmissionUpdated", {
      tokenId: tokenId.toString(),
      newIpfsHash,
      newAuthorshipScore,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  // Listen for role changes
  contract.on("RoleGranted", (role, account, sender, event) => {
    console.log("\n👤 Role Granted:");
    console.log(`  Role: ${role}`);
    console.log(`  Account: ${account}`);
    console.log(`  Granted by: ${sender}`);
    console.log(`  Block: ${event.blockNumber}`);
    
    logEvent("RoleGranted", {
      role,
      account,
      sender,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  contract.on("RoleRevoked", (role, account, sender, event) => {
    console.log("\n👤 Role Revoked:");
    console.log(`  Role: ${role}`);
    console.log(`  Account: ${account}`);
    console.log(`  Revoked by: ${sender}`);
    console.log(`  Block: ${event.blockNumber}`);
    
    logEvent("RoleRevoked", {
      role,
      account,
      sender,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  // Listen for pause/unpause events
  contract.on("Paused", (account, event) => {
    console.log("\n⏸️ Contract Paused:");
    console.log(`  Paused by: ${account}`);
    console.log(`  Block: ${event.blockNumber}`);
    
    logEvent("Paused", {
      account,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  contract.on("Unpaused", (account, event) => {
    console.log("\n▶️ Contract Unpaused:");
    console.log(`  Unpaused by: ${account}`);
    console.log(`  Block: ${event.blockNumber}`);
    
    logEvent("Unpaused", {
      account,
      blockNumber: event.blockNumber,
      transactionHash: event.transactionHash,
      timestamp: new Date().toISOString(),
    });
  });

  console.log("Event monitoring started. Press Ctrl+C to stop.");
  
  // Keep the process running
  process.stdin.resume();
}

function logEvent(eventType, data) {
  const logsDir = path.join(__dirname, "..", "logs");
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }
  
  const logFile = path.join(logsDir, `events-${hre.network.name}.jsonl`);
  const logEntry = JSON.stringify({ eventType, ...data }) + "\n";
  
  fs.appendFileSync(logFile, logEntry);
}

async function getHistoricalEvents(fromBlock = 0, toBlock = "latest") {
  console.log(`Fetching historical events from block ${fromBlock} to ${toBlock}...`);
  
  const { contract } = await loadContract();
  
  // Get all events
  const events = await contract.queryFilter("*", fromBlock, toBlock);
  
  console.log(`Found ${events.length} events`);
  
  const eventSummary = {};
  
  for (const event of events) {
    const eventName = event.eventName || event.fragment?.name || "Unknown";
    
    if (!eventSummary[eventName]) {
      eventSummary[eventName] = 0;
    }
    eventSummary[eventName]++;
    
    console.log(`\nEvent: ${eventName}`);
    console.log(`  Block: ${event.blockNumber}`);
    console.log(`  Transaction: ${event.transactionHash}`);
    console.log(`  Args:`, event.args);
  }
  
  console.log("\nEvent Summary:");
  for (const [eventName, count] of Object.entries(eventSummary)) {
    console.log(`  ${eventName}: ${count}`);
  }
  
  return events;
}

async function getContractMetrics() {
  const { contract } = await loadContract();
  
  console.log("Fetching contract metrics...");
  
  const totalSupply = await contract.totalSupply();
  const isPaused = await contract.paused();
  const contractAddress = await contract.getAddress();
  
  // Get recent events for activity metrics
  const currentBlock = await ethers.provider.getBlockNumber();
  const fromBlock = Math.max(0, currentBlock - 1000); // Last 1000 blocks
  
  const recentEvents = await contract.queryFilter("*", fromBlock, "latest");
  
  const metrics = {
    contractAddress,
    totalSupply: totalSupply.toString(),
    isPaused,
    currentBlock,
    recentActivity: {
      totalEvents: recentEvents.length,
      blockRange: `${fromBlock} - ${currentBlock}`,
      eventsPerBlock: (recentEvents.length / 1000).toFixed(4),
    },
  };
  
  console.log("Contract Metrics:");
  console.log(JSON.stringify(metrics, null, 2));
  
  return metrics;
}

async function checkContractHealth() {
  console.log("Performing contract health check...");
  
  const { contract } = await loadContract();
  
  const health = {
    timestamp: new Date().toISOString(),
    network: hre.network.name,
    status: "healthy",
    issues: [],
  };
  
  try {
    // Check if contract is responsive
    const totalSupply = await contract.totalSupply();
    console.log(`✅ Contract responsive - Total supply: ${totalSupply}`);
    
    // Check if paused
    const isPaused = await contract.paused();
    if (isPaused) {
      health.issues.push("Contract is paused");
      console.log("⚠️ Contract is paused");
    } else {
      console.log("✅ Contract is not paused");
    }
    
    // Check recent activity
    const currentBlock = await ethers.provider.getBlockNumber();
    const recentEvents = await contract.queryFilter("*", currentBlock - 100, "latest");
    
    if (recentEvents.length === 0) {
      health.issues.push("No recent activity (last 100 blocks)");
      console.log("⚠️ No recent activity");
    } else {
      console.log(`✅ Recent activity: ${recentEvents.length} events in last 100 blocks`);
    }
    
    // Check gas prices
    const gasPrice = await ethers.provider.getFeeData();
    console.log(`ℹ️ Current gas price: ${ethers.formatUnits(gasPrice.gasPrice, "gwei")} gwei`);
    
    if (health.issues.length === 0) {
      console.log("✅ Contract health check passed");
    } else {
      health.status = "warning";
      console.log("⚠️ Contract health check completed with warnings");
    }
    
  } catch (error) {
    health.status = "error";
    health.issues.push(`Contract error: ${error.message}`);
    console.log("❌ Contract health check failed:", error.message);
  }
  
  return health;
}

// CLI interface
async function main() {
  const command = process.argv[2];
  const arg1 = process.argv[3];
  const arg2 = process.argv[4];
  
  try {
    switch (command) {
      case "monitor":
        await monitorEvents(arg1);
        break;
        
      case "history":
        await getHistoricalEvents(
          arg1 ? parseInt(arg1) : 0,
          arg2 || "latest"
        );
        break;
        
      case "metrics":
        await getContractMetrics();
        break;
        
      case "health":
        await checkContractHealth();
        break;
        
      default:
        console.log("Available commands:");
        console.log("  monitor [fromBlock]           - Monitor real-time events");
        console.log("  history [fromBlock] [toBlock] - Get historical events");
        console.log("  metrics                       - Get contract metrics");
        console.log("  health                        - Perform health check");
    }
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

// Export functions for use in other scripts
module.exports = {
  monitorEvents,
  getHistoricalEvents,
  getContractMetrics,
  checkContractHealth,
  loadContract,
};

// Run CLI if called directly
if (require.main === module) {
  main()
    .then(() => {
      if (process.argv[2] !== "monitor") {
        process.exit(0);
      }
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}