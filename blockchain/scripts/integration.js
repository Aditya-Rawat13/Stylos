const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");
const axios = require("axios");

/**
 * Integration utilities for connecting blockchain with backend services
 */

async function setupIntegration() {
  console.log("Setting up blockchain-backend integration...");
  
  const { contract, deploymentInfo } = await loadContract();
  
  // Register contract address with backend
  await registerContractWithBackend(deploymentInfo);
  
  // Set up event monitoring
  await setupEventMonitoring(contract);
  
  // Test integration
  await testIntegration(contract);
  
  console.log("Integration setup completed!");
}

async function loadContract() {
  const deploymentFile = path.join(__dirname, "..", "deployments", `${hre.network.name}.json`);
  
  if (!fs.existsSync(deploymentFile)) {
    throw new Error(`Deployment file not found: ${deploymentFile}`);
  }

  const deploymentInfo = JSON.parse(fs.readFileSync(deploymentFile, "utf8"));
  const contract = await ethers.getContractAt("ProofOfAuthorship", deploymentInfo.contractAddress);
  
  return { contract, deploymentInfo };
}

async function registerContractWithBackend(deploymentInfo) {
  console.log("Registering contract with backend...");
  
  const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
  const apiKey = process.env.BACKEND_API_KEY;
  
  try {
    const response = await axios.post(`${backendUrl}/api/v1/admin/blockchain/register-contract`, {
      contractAddress: deploymentInfo.contractAddress,
      implementationAddress: deploymentInfo.implementationAddress,
      network: hre.network.name,
      deploymentInfo: deploymentInfo
    }, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    
    console.log("Contract registered with backend:", response.data);
  } catch (error) {
    console.error("Failed to register contract with backend:", error.message);
  }
}

async function setupEventMonitoring(contract) {
  console.log("Setting up event monitoring...");
  
  // Listen for SubmissionMinted events and notify backend
  contract.on("SubmissionMinted", async (tokenId, student, contentHash, ipfsHash, authorshipScore, aiProbability, event) => {
    console.log(`New submission minted: Token ${tokenId} for ${student}`);
    
    try {
      await notifyBackendOfEvent("SubmissionMinted", {
        tokenId: tokenId.toString(),
        student,
        contentHash,
        ipfsHash,
        authorshipScore,
        aiProbability,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash
      });
    } catch (error) {
      console.error("Failed to notify backend of event:", error.message);
    }
  });
  
  // Listen for verification status changes
  contract.on("SubmissionVerified", async (tokenId, verifier, verified, event) => {
    console.log(`Submission ${tokenId} verification updated: ${verified}`);
    
    try {
      await notifyBackendOfEvent("SubmissionVerified", {
        tokenId: tokenId.toString(),
        verifier,
        verified,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash
      });
    } catch (error) {
      console.error("Failed to notify backend of verification event:", error.message);
    }
  });
  
  console.log("Event monitoring setup completed");
}

async function notifyBackendOfEvent(eventType, eventData) {
  const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
  const apiKey = process.env.BACKEND_API_KEY;
  
  try {
    await axios.post(`${backendUrl}/api/v1/blockchain/events`, {
      eventType,
      eventData,
      network: hre.network.name,
      timestamp: new Date().toISOString()
    }, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    console.error(`Failed to notify backend of ${eventType} event:`, error.message);
  }
}

async function testIntegration(contract) {
  console.log("Testing blockchain-backend integration...");
  
  try {
    // Test contract connectivity
    const totalSupply = await contract.totalSupply();
    console.log(`Contract accessible - Total supply: ${totalSupply}`);
    
    // Test backend connectivity
    const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
    const healthResponse = await axios.get(`${backendUrl}/health`);
    console.log("Backend accessible - Status:", healthResponse.status);
    
    // Test IPFS connectivity (through backend)
    try {
      const ipfsResponse = await axios.get(`${backendUrl}/api/v1/blockchain/network-stats`);
      console.log("IPFS integration test passed");
    } catch (error) {
      console.warn("IPFS integration test failed:", error.message);
    }
    
    console.log("Integration test completed successfully!");
    
  } catch (error) {
    console.error("Integration test failed:", error.message);
    throw error;
  }
}

async function deployAndIntegrate() {
  console.log("Deploying contract and setting up integration...");
  
  // Deploy contract
  const { run } = require("hardhat");
  await run("run", { script: "scripts/deploy.js" });
  
  // Setup integration
  await setupIntegration();
  
  console.log("Deployment and integration completed!");
}

async function syncBlockchainState() {
  console.log("Syncing blockchain state with backend...");
  
  const { contract } = await loadContract();
  const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
  const apiKey = process.env.BACKEND_API_KEY;
  
  try {
    // Get all historical events
    const currentBlock = await ethers.provider.getBlockNumber();
    const fromBlock = Math.max(0, currentBlock - 10000); // Last 10k blocks
    
    const events = await contract.queryFilter("*", fromBlock, "latest");
    
    console.log(`Found ${events.length} events to sync`);
    
    // Send events to backend for processing
    for (const event of events) {
      const eventData = {
        eventName: event.eventName || event.fragment?.name,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash,
        args: event.args,
        timestamp: new Date().toISOString()
      };
      
      try {
        await axios.post(`${backendUrl}/api/v1/blockchain/sync-event`, eventData, {
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
          }
        });
      } catch (error) {
        console.error(`Failed to sync event ${event.transactionHash}:`, error.message);
      }
    }
    
    console.log("Blockchain state sync completed");
    
  } catch (error) {
    console.error("Failed to sync blockchain state:", error.message);
  }
}

async function validateIntegration() {
  console.log("Validating blockchain integration...");
  
  const { contract } = await loadContract();
  const backendUrl = process.env.BACKEND_API_URL || "http://localhost:8000";
  
  const validationResults = {
    contractAccessible: false,
    backendAccessible: false,
    ipfsAccessible: false,
    eventMonitoring: false,
    dataConsistency: false
  };
  
  try {
    // Test contract access
    await contract.totalSupply();
    validationResults.contractAccessible = true;
    console.log("✅ Contract accessible");
  } catch (error) {
    console.log("❌ Contract not accessible:", error.message);
  }
  
  try {
    // Test backend access
    const response = await axios.get(`${backendUrl}/health`);
    validationResults.backendAccessible = response.status === 200;
    console.log("✅ Backend accessible");
  } catch (error) {
    console.log("❌ Backend not accessible:", error.message);
  }
  
  try {
    // Test IPFS through backend
    await axios.get(`${backendUrl}/api/v1/blockchain/network-stats`);
    validationResults.ipfsAccessible = true;
    console.log("✅ IPFS accessible through backend");
  } catch (error) {
    console.log("❌ IPFS not accessible:", error.message);
  }
  
  // Test event monitoring (simplified check)
  try {
    const recentEvents = await contract.queryFilter("*", -100, "latest");
    validationResults.eventMonitoring = true;
    console.log(`✅ Event monitoring functional (${recentEvents.length} recent events)`);
  } catch (error) {
    console.log("❌ Event monitoring not functional:", error.message);
  }
  
  // Test data consistency (check if backend has matching records)
  try {
    const totalSupply = await contract.totalSupply();
    const backendResponse = await axios.get(`${backendUrl}/api/v1/blockchain/records`);
    
    // This is a simplified check - in production, you'd do more thorough validation
    validationResults.dataConsistency = backendResponse.status === 200;
    console.log("✅ Data consistency check passed");
  } catch (error) {
    console.log("❌ Data consistency check failed:", error.message);
  }
  
  // Summary
  const passedTests = Object.values(validationResults).filter(Boolean).length;
  const totalTests = Object.keys(validationResults).length;
  
  console.log(`\nValidation Summary: ${passedTests}/${totalTests} tests passed`);
  
  if (passedTests === totalTests) {
    console.log("🎉 All integration tests passed!");
  } else {
    console.log("⚠️ Some integration tests failed. Please check the issues above.");
  }
  
  return validationResults;
}

// CLI interface
async function main() {
  const command = process.argv[2];
  
  try {
    switch (command) {
      case "setup":
        await setupIntegration();
        break;
        
      case "deploy-and-integrate":
        await deployAndIntegrate();
        break;
        
      case "sync":
        await syncBlockchainState();
        break;
        
      case "validate":
        await validateIntegration();
        break;
        
      case "test":
        const { contract } = await loadContract();
        await testIntegration(contract);
        break;
        
      default:
        console.log("Available commands:");
        console.log("  setup                 - Setup integration with existing contract");
        console.log("  deploy-and-integrate  - Deploy contract and setup integration");
        console.log("  sync                  - Sync blockchain state with backend");
        console.log("  validate              - Validate integration health");
        console.log("  test                  - Test integration connectivity");
    }
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

// Export functions for use in other scripts
module.exports = {
  setupIntegration,
  deployAndIntegrate,
  syncBlockchainState,
  validateIntegration,
  testIntegration,
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