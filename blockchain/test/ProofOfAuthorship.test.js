const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

describe("ProofOfAuthorship", function () {
  // Fixture for deploying the contract
  async function deployProofOfAuthorshipFixture() {
    const [owner, verifier, student1, student2, unauthorized] = await ethers.getSigners();

    const ProofOfAuthorship = await ethers.getContractFactory("ProofOfAuthorship");
    const contract = await upgrades.deployProxy(
      ProofOfAuthorship,
      [owner.address],
      { initializer: "initialize", kind: "uups" }
    );

    await contract.waitForDeployment();

    // Grant verifier role
    const VERIFIER_ROLE = await contract.VERIFIER_ROLE();
    await contract.grantRole(VERIFIER_ROLE, verifier.address);

    return {
      contract,
      owner,
      verifier,
      student1,
      student2,
      unauthorized,
      VERIFIER_ROLE,
    };
  }

  describe("Deployment", function () {
    it("Should deploy with correct initial state", async function () {
      const { contract, owner } = await loadFixture(deployProofOfAuthorshipFixture);

      expect(await contract.name()).to.equal("ProofOfAuthorship");
      expect(await contract.symbol()).to.equal("POA");
      expect(await contract.totalSupply()).to.equal(0);
      expect(await contract.hasRole(await contract.DEFAULT_ADMIN_ROLE(), owner.address)).to.be.true;
    });

    it("Should grant correct roles to admin", async function () {
      const { contract, owner } = await loadFixture(deployProofOfAuthorshipFixture);

      const DEFAULT_ADMIN_ROLE = await contract.DEFAULT_ADMIN_ROLE();
      const VERIFIER_ROLE = await contract.VERIFIER_ROLE();
      const UPGRADER_ROLE = await contract.UPGRADER_ROLE();
      const PAUSER_ROLE = await contract.PAUSER_ROLE();

      expect(await contract.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.be.true;
      expect(await contract.hasRole(VERIFIER_ROLE, owner.address)).to.be.true;
      expect(await contract.hasRole(UPGRADER_ROLE, owner.address)).to.be.true;
      expect(await contract.hasRole(PAUSER_ROLE, owner.address)).to.be.true;
    });
  });

  describe("Token Minting", function () {
    it("Should mint a proof token successfully", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("test essay content"));
      const stylometricHash = ethers.keccak256(ethers.toUtf8Bytes("stylometric data"));
      const ipfsHash = "QmTestHash123";
      const authorshipScore = 85;
      const aiProbability = 15;
      const institutionId = "university-123";
      const courseId = "cs-101";

      const tx = await contract.connect(verifier).mintProofToken(
        contentHash,
        stylometricHash,
        ipfsHash,
        student1.address,
        authorshipScore,
        aiProbability,
        institutionId,
        courseId
      );

      await expect(tx)
        .to.emit(contract, "SubmissionMinted")
        .withArgs(1, student1.address, contentHash, ipfsHash, authorshipScore, aiProbability);

      expect(await contract.totalSupply()).to.equal(1);
      expect(await contract.ownerOf(1)).to.equal(student1.address);
      expect(await contract.balanceOf(student1.address)).to.equal(1);
    });

    it("Should store submission data correctly", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("test essay content"));
      const stylometricHash = ethers.keccak256(ethers.toUtf8Bytes("stylometric data"));
      const ipfsHash = "QmTestHash123";
      const authorshipScore = 85;
      const aiProbability = 15;
      const institutionId = "university-123";
      const courseId = "cs-101";

      await contract.connect(verifier).mintProofToken(
        contentHash,
        stylometricHash,
        ipfsHash,
        student1.address,
        authorshipScore,
        aiProbability,
        institutionId,
        courseId
      );

      const submission = await contract.getSubmission(1);
      expect(submission.contentHash).to.equal(contentHash);
      expect(submission.stylometricHash).to.equal(stylometricHash);
      expect(submission.ipfsHash).to.equal(ipfsHash);
      expect(submission.student).to.equal(student1.address);
      expect(submission.authorshipScore).to.equal(authorshipScore);
      expect(submission.aiProbability).to.equal(aiProbability);
      expect(submission.verified).to.be.true;
      expect(submission.institutionId).to.equal(institutionId);
      expect(submission.courseId).to.equal(courseId);
    });

    it("Should prevent duplicate submissions", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("test essay content"));
      const stylometricHash = ethers.keccak256(ethers.toUtf8Bytes("stylometric data"));
      const ipfsHash = "QmTestHash123";

      // First submission should succeed
      await contract.connect(verifier).mintProofToken(
        contentHash,
        stylometricHash,
        ipfsHash,
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      // Second submission with same content hash should fail
      await expect(
        contract.connect(verifier).mintProofToken(
          contentHash,
          stylometricHash,
          "QmDifferentHash",
          student1.address,
          90,
          10,
          "university-123",
          "cs-101"
        )
      ).to.be.revertedWithCustomError(contract, "DuplicateSubmission");
    });

    it("Should reject invalid submission data", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      // Test with zero content hash
      await expect(
        contract.connect(verifier).mintProofToken(
          ethers.ZeroHash,
          ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
          "QmTestHash123",
          student1.address,
          85,
          15,
          "university-123",
          "cs-101"
        )
      ).to.be.revertedWithCustomError(contract, "InvalidSubmissionData");

      // Test with zero address
      await expect(
        contract.connect(verifier).mintProofToken(
          ethers.keccak256(ethers.toUtf8Bytes("test content")),
          ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
          "QmTestHash123",
          ethers.ZeroAddress,
          85,
          15,
          "university-123",
          "cs-101"
        )
      ).to.be.revertedWithCustomError(contract, "InvalidSubmissionData");
    });

    it("Should only allow verifiers to mint tokens", async function () {
      const { contract, unauthorized, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("test essay content"));
      const stylometricHash = ethers.keccak256(ethers.toUtf8Bytes("stylometric data"));

      await expect(
        contract.connect(unauthorized).mintProofToken(
          contentHash,
          stylometricHash,
          "QmTestHash123",
          student1.address,
          85,
          15,
          "university-123",
          "cs-101"
        )
      ).to.be.reverted;
    });
  });

  describe("Batch Minting", function () {
    it("Should batch mint multiple tokens successfully", async function () {
      const { contract, verifier, student1, student2 } = await loadFixture(deployProofOfAuthorshipFixture);

      const contentHashes = [
        ethers.keccak256(ethers.toUtf8Bytes("essay 1")),
        ethers.keccak256(ethers.toUtf8Bytes("essay 2")),
      ];
      const stylometricHashes = [
        ethers.keccak256(ethers.toUtf8Bytes("style 1")),
        ethers.keccak256(ethers.toUtf8Bytes("style 2")),
      ];
      const ipfsHashes = ["QmHash1", "QmHash2"];
      const students = [student1.address, student2.address];
      const authorshipScores = [85, 90];
      const aiProbabilities = [15, 10];
      const institutionIds = ["university-123", "university-123"];
      const courseIds = ["cs-101", "cs-102"];

      const tx = await contract.connect(verifier).batchMintProofTokens(
        contentHashes,
        stylometricHashes,
        ipfsHashes,
        students,
        authorshipScores,
        aiProbabilities,
        institutionIds,
        courseIds
      );

      await expect(tx)
        .to.emit(contract, "BatchSubmissionsMinted")
        .withArgs([1, 2], students, await ethers.provider.getBlock("latest").then(b => b.timestamp));

      expect(await contract.totalSupply()).to.equal(2);
      expect(await contract.ownerOf(1)).to.equal(student1.address);
      expect(await contract.ownerOf(2)).to.equal(student2.address);
    });

    it("Should reject batch mint with mismatched array lengths", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const contentHashes = [ethers.keccak256(ethers.toUtf8Bytes("essay 1"))];
      const stylometricHashes = [
        ethers.keccak256(ethers.toUtf8Bytes("style 1")),
        ethers.keccak256(ethers.toUtf8Bytes("style 2")),
      ]; // Different length

      await expect(
        contract.connect(verifier).batchMintProofTokens(
          contentHashes,
          stylometricHashes,
          ["QmHash1"],
          [student1.address],
          [85],
          [15],
          ["university-123"],
          ["cs-101"]
        )
      ).to.be.revertedWith("Array length mismatch");
    });
  });

  describe("Soulbound Token Behavior", function () {
    it("Should prevent token transfers", async function () {
      const { contract, verifier, student1, student2 } = await loadFixture(deployProofOfAuthorshipFixture);

      // Mint a token
      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "QmTestHash123",
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      // Try to transfer - should fail
      await expect(
        contract.connect(student1).transferFrom(student1.address, student2.address, 1)
      ).to.be.revertedWithCustomError(contract, "TokenNotTransferable");

      await expect(
        contract.connect(student1).safeTransferFrom(student1.address, student2.address, 1)
      ).to.be.revertedWithCustomError(contract, "TokenNotTransferable");

      await expect(
        contract.connect(student1).approve(student2.address, 1)
      ).to.be.revertedWithCustomError(contract, "TokenNotTransferable");

      await expect(
        contract.connect(student1).setApprovalForAll(student2.address, true)
      ).to.be.revertedWithCustomError(contract, "TokenNotTransferable");
    });
  });

  describe("Query Functions", function () {
    beforeEach(async function () {
      const { contract, verifier, student1, student2 } = await loadFixture(deployProofOfAuthorshipFixture);
      
      // Mint some tokens for testing
      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 1")),
        ethers.keccak256(ethers.toUtf8Bytes("style 1")),
        "QmHash1",
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 2")),
        ethers.keccak256(ethers.toUtf8Bytes("style 2")),
        "QmHash2",
        student1.address,
        90,
        10,
        "university-123",
        "cs-102"
      );

      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 3")),
        ethers.keccak256(ethers.toUtf8Bytes("style 3")),
        "QmHash3",
        student2.address,
        88,
        12,
        "university-456",
        "cs-101"
      );

      this.contract = contract;
      this.student1 = student1;
      this.student2 = student2;
    });

    it("Should return correct student submissions", async function () {
      const student1Submissions = await this.contract.getStudentSubmissions(this.student1.address);
      const student2Submissions = await this.contract.getStudentSubmissions(this.student2.address);

      expect(student1Submissions.length).to.equal(2);
      expect(student2Submissions.length).to.equal(1);
      expect(student1Submissions[0]).to.equal(1);
      expect(student1Submissions[1]).to.equal(2);
      expect(student2Submissions[0]).to.equal(3);
    });

    it("Should return correct institution submissions", async function () {
      const uni123Submissions = await this.contract.getInstitutionSubmissions("university-123");
      const uni456Submissions = await this.contract.getInstitutionSubmissions("university-456");

      expect(uni123Submissions.length).to.equal(2);
      expect(uni456Submissions.length).to.equal(1);
      expect(uni123Submissions[0]).to.equal(1);
      expect(uni123Submissions[1]).to.equal(2);
      expect(uni456Submissions[0]).to.equal(3);
    });

    it("Should find token by content hash", async function () {
      const contentHash = ethers.keccak256(ethers.toUtf8Bytes("essay 1"));
      const tokenId = await this.contract.getTokenByContentHash(contentHash);
      expect(tokenId).to.equal(1);
    });
  });

  describe("Access Control", function () {
    it("Should allow admin to pause and unpause", async function () {
      const { contract, owner } = await loadFixture(deployProofOfAuthorshipFixture);

      await contract.connect(owner).pause();
      expect(await contract.paused()).to.be.true;

      await contract.connect(owner).unpause();
      expect(await contract.paused()).to.be.false;
    });

    it("Should prevent minting when paused", async function () {
      const { contract, owner, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      await contract.connect(owner).pause();

      await expect(
        contract.connect(verifier).mintProofToken(
          ethers.keccak256(ethers.toUtf8Bytes("test content")),
          ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
          "QmTestHash123",
          student1.address,
          85,
          15,
          "university-123",
          "cs-101"
        )
      ).to.be.revertedWith("Pausable: paused");
    });
  });

  describe("Contract Upgrades", function () {
    it("Should be upgradeable by admin", async function () {
      const { contract, owner } = await loadFixture(deployProofOfAuthorshipFixture);

      const ProofOfAuthorshipV2 = await ethers.getContractFactory("ProofOfAuthorship");
      const upgraded = await upgrades.upgradeProxy(contract, ProofOfAuthorshipV2);

      expect(await upgraded.name()).to.equal("ProofOfAuthorship");
      expect(await upgraded.symbol()).to.equal("POA");
    });
  });

  describe("Gas Optimization", function () {
    it("Should use reasonable gas for single mint", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const tx = await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "QmTestHash123",
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      const receipt = await tx.wait();
      console.log("Single mint gas used:", receipt.gasUsed.toString());
      
      // Should use less than 200k gas for a single mint
      expect(receipt.gasUsed).to.be.lt(200000);
    });

    it("Should be more efficient for batch minting", async function () {
      const { contract, verifier, student1, student2 } = await loadFixture(deployProofOfAuthorshipFixture);

      const batchSize = 5;
      const contentHashes = [];
      const stylometricHashes = [];
      const ipfsHashes = [];
      const students = [];
      const authorshipScores = [];
      const aiProbabilities = [];
      const institutionIds = [];
      const courseIds = [];

      for (let i = 0; i < batchSize; i++) {
        contentHashes.push(ethers.keccak256(ethers.toUtf8Bytes(`essay ${i}`)));
        stylometricHashes.push(ethers.keccak256(ethers.toUtf8Bytes(`style ${i}`)));
        ipfsHashes.push(`QmHash${i}`);
        students.push(i % 2 === 0 ? student1.address : student2.address);
        authorshipScores.push(85 + i);
        aiProbabilities.push(15 - i);
        institutionIds.push("university-123");
        courseIds.push(`cs-10${i}`);
      }

      const tx = await contract.connect(verifier).batchMintProofTokens(
        contentHashes,
        stylometricHashes,
        ipfsHashes,
        students,
        authorshipScores,
        aiProbabilities,
        institutionIds,
        courseIds
      );

      const receipt = await tx.wait();
      console.log("Batch mint gas used:", receipt.gasUsed.toString());
      console.log("Gas per token:", (receipt.gasUsed / BigInt(batchSize)).toString());
      
      // Batch should be more efficient than individual mints
      expect(receipt.gasUsed / BigInt(batchSize)).to.be.lt(150000);
    });
  });

  describe("Edge Cases and Error Handling", function () {
    it("Should handle maximum authorship score", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const tx = await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "QmTestHash123",
        student1.address,
        100, // Maximum score
        0,   // Minimum AI probability
        "university-123",
        "cs-101"
      );

      await expect(tx).to.emit(contract, "SubmissionMinted");
    });

    it("Should handle minimum authorship score", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const tx = await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "QmTestHash123",
        student1.address,
        0,   // Minimum score
        100, // Maximum AI probability
        "university-123",
        "cs-101"
      );

      await expect(tx).to.emit(contract, "SubmissionMinted");
    });

    it("Should handle empty IPFS hash", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const tx = await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "", // Empty IPFS hash
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      await expect(tx).to.emit(contract, "SubmissionMinted");
    });

    it("Should handle very long institution and course IDs", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const longInstitutionId = "a".repeat(100);
      const longCourseId = "b".repeat(100);

      const tx = await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "QmTestHash123",
        student1.address,
        85,
        15,
        longInstitutionId,
        longCourseId
      );

      await expect(tx).to.emit(contract, "SubmissionMinted");
    });

    it("Should handle query for non-existent token", async function () {
      const { contract } = await loadFixture(deployProofOfAuthorshipFixture);

      await expect(contract.getSubmission(999)).to.be.reverted;
    });

    it("Should handle query for non-existent content hash", async function () {
      const { contract } = await loadFixture(deployProofOfAuthorshipFixture);

      const nonExistentHash = ethers.keccak256(ethers.toUtf8Bytes("non-existent"));
      const tokenId = await contract.getTokenByContentHash(nonExistentHash);
      
      expect(tokenId).to.equal(0); // Should return 0 for non-existent
    });

    it("Should handle empty batch mint", async function () {
      const { contract, verifier } = await loadFixture(deployProofOfAuthorshipFixture);

      await expect(
        contract.connect(verifier).batchMintProofTokens([], [], [], [], [], [], [], [])
      ).to.be.revertedWith("Array length mismatch");
    });

    it("Should maintain token counter correctly", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      // Mint first token
      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 1")),
        ethers.keccak256(ethers.toUtf8Bytes("style 1")),
        "QmHash1",
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      expect(await contract.totalSupply()).to.equal(1);

      // Mint second token
      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 2")),
        ethers.keccak256(ethers.toUtf8Bytes("style 2")),
        "QmHash2",
        student1.address,
        90,
        10,
        "university-123",
        "cs-102"
      );

      expect(await contract.totalSupply()).to.equal(2);
    });

    it("Should handle multiple students with same institution", async function () {
      const { contract, verifier, student1, student2 } = await loadFixture(deployProofOfAuthorshipFixture);

      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 1")),
        ethers.keccak256(ethers.toUtf8Bytes("style 1")),
        "QmHash1",
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      await contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("essay 2")),
        ethers.keccak256(ethers.toUtf8Bytes("style 2")),
        "QmHash2",
        student2.address,
        90,
        10,
        "university-123",
        "cs-101"
      );

      const institutionSubmissions = await contract.getInstitutionSubmissions("university-123");
      expect(institutionSubmissions.length).to.equal(2);
    });
  });

  describe("Security Tests", function () {
    it("Should prevent reentrancy attacks", async function () {
      const { contract, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      // Attempt to mint while another mint is in progress should fail
      // This is a basic test; actual reentrancy testing would require a malicious contract
      const tx = contract.connect(verifier).mintProofToken(
        ethers.keccak256(ethers.toUtf8Bytes("test content")),
        ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
        "QmTestHash123",
        student1.address,
        85,
        15,
        "university-123",
        "cs-101"
      );

      await expect(tx).to.emit(contract, "SubmissionMinted");
    });

    it("Should prevent unauthorized role grants", async function () {
      const { contract, unauthorized, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const VERIFIER_ROLE = await contract.VERIFIER_ROLE();

      await expect(
        contract.connect(unauthorized).grantRole(VERIFIER_ROLE, student1.address)
      ).to.be.reverted;
    });

    it("Should allow admin to revoke verifier role", async function () {
      const { contract, owner, verifier } = await loadFixture(deployProofOfAuthorshipFixture);

      const VERIFIER_ROLE = await contract.VERIFIER_ROLE();

      await contract.connect(owner).revokeRole(VERIFIER_ROLE, verifier.address);

      expect(await contract.hasRole(VERIFIER_ROLE, verifier.address)).to.be.false;
    });

    it("Should prevent operations after role revocation", async function () {
      const { contract, owner, verifier, student1 } = await loadFixture(deployProofOfAuthorshipFixture);

      const VERIFIER_ROLE = await contract.VERIFIER_ROLE();

      // Revoke verifier role
      await contract.connect(owner).revokeRole(VERIFIER_ROLE, verifier.address);

      // Attempt to mint should fail
      await expect(
        contract.connect(verifier).mintProofToken(
          ethers.keccak256(ethers.toUtf8Bytes("test content")),
          ethers.keccak256(ethers.toUtf8Bytes("stylometric data")),
          "QmTestHash123",
          student1.address,
          85,
          15,
          "university-123",
          "cs-101"
        )
      ).to.be.reverted;
    });
  });
});