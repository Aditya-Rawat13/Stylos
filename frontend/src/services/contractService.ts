import { Contract, BrowserProvider } from 'ethers';
import { web3Service } from './web3Service';

// Contract ABI - will be loaded from public/contracts/ProofOfAuthorship.json
let contractABI: any[] = [];
let contractAddress: string = '';

// Load contract configuration
export const loadContractConfig = async (): Promise<void> => {
  try {
    const response = await fetch('/contracts/ProofOfAuthorship.json');
    if (!response.ok) {
      throw new Error('Failed to load contract configuration');
    }
    const config = await response.json();
    contractABI = config.abi;
    contractAddress = config.address || process.env.REACT_APP_CONTRACT_ADDRESS || '';
    
    console.log('Contract configuration loaded:', {
      address: contractAddress,
      abiLength: contractABI.length,
    });
  } catch (error) {
    console.error('Failed to load contract configuration:', error);
    // Fallback to environment variable
    contractAddress = process.env.REACT_APP_CONTRACT_ADDRESS || '';
  }
};

export interface SubmissionData {
  contentHash: string;
  stylometricHash: string;
  ipfsHash: string;
  student: string;
  timestamp: number;
  authorshipScore: number;
  aiProbability: number;
  verified: boolean;
  institutionId: string;
  courseId: string;
}

export interface MintTokenParams {
  contentHash: string;
  stylometricHash: string;
  ipfsHash: string;
  studentAddress: string;
  authorshipScore: number;
  aiProbability: number;
  institutionId: string;
  courseId: string;
}

class ContractService {
  private contract: Contract | null = null;

  /**
   * Initialize contract instance
   */
  async initializeContract(): Promise<void> {
    if (!contractAddress || contractABI.length === 0) {
      await loadContractConfig();
    }

    if (!contractAddress) {
      throw new Error('Contract address not configured');
    }

    const provider = web3Service.getProvider();
    const signer = web3Service.getSigner();

    if (!provider) {
      throw new Error('Web3 provider not initialized');
    }

    // Use signer if available (for write operations), otherwise use provider (read-only)
    this.contract = new Contract(contractAddress, contractABI, signer || provider);
    console.log('Contract initialized at:', contractAddress);
  }

  /**
   * Get contract instance
   */
  getContract(): Contract {
    if (!this.contract) {
      throw new Error('Contract not initialized. Call initializeContract() first.');
    }
    return this.contract;
  }

  /**
   * Mint a new proof-of-authorship token
   */
  async mintProofToken(params: MintTokenParams): Promise<{ tokenId: string; txHash: string }> {
    if (!this.contract) {
      await this.initializeContract();
    }

    try {
      const tx = await this.contract!.mintProofToken(
        params.contentHash,
        params.stylometricHash,
        params.ipfsHash,
        params.studentAddress,
        params.authorshipScore,
        params.aiProbability,
        params.institutionId,
        params.courseId
      );

      console.log('Transaction sent:', tx.hash);
      const receipt = await tx.wait();
      console.log('Transaction confirmed:', receipt);

      // Extract token ID from event logs
      const event = receipt.logs.find((log: any) => {
        try {
          const parsed = this.contract!.interface.parseLog(log);
          return parsed?.name === 'SubmissionMinted';
        } catch {
          return false;
        }
      });

      let tokenId = '0';
      if (event) {
        const parsed = this.contract!.interface.parseLog(event);
        tokenId = parsed?.args?.tokenId?.toString() || '0';
      }

      return {
        tokenId,
        txHash: receipt.hash,
      };
    } catch (error: any) {
      console.error('Failed to mint token:', error);
      throw new Error(error.message || 'Failed to mint proof token');
    }
  }

  /**
   * Get submission details by token ID
   */
  async getSubmission(tokenId: string): Promise<SubmissionData> {
    if (!this.contract) {
      await this.initializeContract();
    }

    try {
      const submission = await this.contract!.getSubmission(tokenId);

      return {
        contentHash: submission.contentHash,
        stylometricHash: submission.stylometricHash,
        ipfsHash: submission.ipfsHash,
        student: submission.student,
        timestamp: Number(submission.timestamp),
        authorshipScore: submission.authorshipScore,
        aiProbability: submission.aiProbability,
        verified: submission.verified,
        institutionId: submission.institutionId,
        courseId: submission.courseId,
      };
    } catch (error: any) {
      console.error('Failed to get submission:', error);
      throw new Error(error.message || 'Failed to get submission details');
    }
  }

  /**
   * Get all submissions for a student
   */
  async getStudentSubmissions(studentAddress: string): Promise<string[]> {
    if (!this.contract) {
      await this.initializeContract();
    }

    try {
      const tokenIds = await this.contract!.getStudentSubmissions(studentAddress);
      return tokenIds.map((id: bigint) => id.toString());
    } catch (error: any) {
      console.error('Failed to get student submissions:', error);
      throw new Error(error.message || 'Failed to get student submissions');
    }
  }

  /**
   * Get total supply of tokens
   */
  async getTotalSupply(): Promise<number> {
    if (!this.contract) {
      await this.initializeContract();
    }

    try {
      const total = await this.contract!.totalSupply();
      return Number(total);
    } catch (error: any) {
      console.error('Failed to get total supply:', error);
      return 0;
    }
  }

  /**
   * Get token by content hash
   */
  async getTokenByContentHash(contentHash: string): Promise<string> {
    if (!this.contract) {
      await this.initializeContract();
    }

    try {
      const tokenId = await this.contract!.getTokenByContentHash(contentHash);
      return tokenId.toString();
    } catch (error: any) {
      console.error('Failed to get token by content hash:', error);
      throw new Error(error.message || 'Failed to get token by content hash');
    }
  }

  /**
   * Check if a submission exists
   */
  async submissionExists(contentHash: string): Promise<boolean> {
    try {
      const tokenId = await this.getTokenByContentHash(contentHash);
      return tokenId !== '0';
    } catch {
      return false;
    }
  }

  /**
   * Get contract address
   */
  getContractAddress(): string {
    return contractAddress;
  }
}

// Export singleton instance
export const contractService = new ContractService();

// Initialize on module load
loadContractConfig().catch(console.error);
