import { api } from './authService';

export interface BlockchainRecord {
  id: string;
  submissionId: string;
  transactionHash: string;
  blockNumber: number;
  contractAddress: string;
  tokenId: string;
  ipfsHash?: string;
  timestamp: string;
  status: 'PENDING' | 'CONFIRMED' | 'FAILED';
  gasUsed?: number;
  gasPrice?: string;
  networkFee?: string;
}

export interface SoulboundToken {
  tokenId: string;
  submissionId: string;
  submissionTitle: string;
  mintedAt: string;
  transactionHash: string;
  ipfsMetadata: {
    name: string;
    description: string;
    attributes: Array<{
      trait_type: string;
      value: string | number;
    }>;
  };
  verificationProof: {
    authorshipScore: number;
    aiProbability: number;
    duplicateStatus: string;
  };
}

export interface BlockchainPortfolio {
  totalTokens: number;
  totalVerifiedSubmissions: number;
  portfolioValue: {
    academicCredibility: number;
    uniquenessScore: number;
    consistencyRating: number;
  };
  tokens: SoulboundToken[];
  recentActivity: Array<{
    type: 'MINT' | 'VERIFY' | 'UPDATE';
    timestamp: string;
    description: string;
    transactionHash?: string;
  }>;
}

export interface TransactionStatus {
  hash: string;
  status: 'PENDING' | 'CONFIRMED' | 'FAILED';
  confirmations: number;
  blockNumber?: number;
  gasUsed?: number;
  error?: string;
}

export const blockchainService = {
  async getPortfolio(): Promise<BlockchainPortfolio> {
    const response = await api.get('/api/v1/blockchain/portfolio');
    return response.data;
  },

  async getBlockchainRecords(page = 1, limit = 10): Promise<{
    records: BlockchainRecord[];
    total: number;
    page: number;
    totalPages: number;
  }> {
    const response = await api.get(`/api/v1/blockchain/records?page=${page}&limit=${limit}`);
    return response.data;
  },

  async getTokenDetails(tokenId: string): Promise<SoulboundToken> {
    const response = await api.get(`/api/v1/blockchain/tokens/${tokenId}`);
    return response.data;
  },

  async verifyTransaction(transactionHash: string): Promise<TransactionStatus> {
    const response = await api.get(`/api/v1/blockchain/verify/${transactionHash}`);
    return response.data;
  },

  async getTransactionStatus(transactionHash: string): Promise<TransactionStatus> {
    const response = await api.get(`/api/v1/blockchain/status/${transactionHash}`);
    return response.data;
  },

  async retryBlockchainAttestation(submissionId: string): Promise<{ transactionHash: string }> {
    const response = await api.post(`/api/v1/blockchain/retry/${submissionId}`);
    return response.data;
  },

  async getNetworkStats(): Promise<{
    network: string;
    blockHeight: number;
    gasPrice: string;
    networkStatus: 'HEALTHY' | 'CONGESTED' | 'OFFLINE';
    estimatedConfirmationTime: number;
  }> {
    const response = await api.get('/api/v1/blockchain/network-stats');
    return response.data;
  },

  async exportPortfolio(format: 'json' | 'pdf'): Promise<Blob> {
    const response = await api.get(`/api/v1/blockchain/export?format=${format}`, {
      responseType: 'blob',
    });
    return response.data;
  },

  async getIPFSContent(ipfsHash: string): Promise<any> {
    const response = await api.get(`/api/v1/blockchain/ipfs/${ipfsHash}`);
    return response.data;
  },

  // Real-time transaction monitoring
  subscribeToTransactionUpdates(
    transactionHash: string,
    onUpdate: (status: TransactionStatus) => void
  ): EventSource {
    const eventSource = new EventSource(
      `${api.defaults.baseURL}/api/v1/blockchain/stream/${transactionHash}`,
      {
        withCredentials: true,
      }
    );

    eventSource.onmessage = (event) => {
      const status = JSON.parse(event.data);
      onUpdate(status);
    };

    return eventSource;
  },
};