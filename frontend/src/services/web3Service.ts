import { BrowserProvider, JsonRpcSigner, Contract, formatEther, parseEther } from 'ethers';

// Hardhat Local Network Configuration
const HARDHAT_NETWORK = {
  chainId: '0x7A69', // 31337 in hex
  chainName: 'Hardhat Local',
  nativeCurrency: {
    name: 'Ethereum',
    symbol: 'ETH',
    decimals: 18,
  },
  rpcUrls: ['http://127.0.0.1:8545'],
  blockExplorerUrls: [],
};

export interface WalletState {
  connected: boolean;
  address: string | null;
  balance: string | null;
  chainId: bigint | null;
  networkName: string | null;
}

class Web3Service {
  private provider: BrowserProvider | null = null;
  private signer: JsonRpcSigner | null = null;
  private walletState: WalletState = {
    connected: false,
    address: null,
    balance: null,
    chainId: null,
    networkName: null,
  };

  /**
   * Check if MetaMask is installed
   */
  isMetaMaskInstalled(): boolean {
    return typeof window !== 'undefined' && typeof window.ethereum !== 'undefined';
  }

  /**
   * Connect to MetaMask wallet
   */
  async connectWallet(): Promise<WalletState> {
    if (!this.isMetaMaskInstalled()) {
      throw new Error('MetaMask is not installed. Please install MetaMask to continue.');
    }

    try {
      // Request account access
      const accounts = await window.ethereum.request({
        method: 'eth_requestAccounts',
      });

      if (!accounts || accounts.length === 0) {
        throw new Error('No accounts found. Please unlock MetaMask.');
      }

      // Initialize provider and signer
      this.provider = new BrowserProvider(window.ethereum);
      this.signer = await this.provider.getSigner();

      // Get network info
      const network = await this.provider.getNetwork();
      const address = await this.signer.getAddress();
      const balance = await this.provider.getBalance(address);

      // Check if on correct network
      if (network.chainId !== BigInt(31337)) {
        await this.switchToHardhatNetwork();
      }

      // Update wallet state
      this.walletState = {
        connected: true,
        address,
        balance: formatEther(balance),
        chainId: network.chainId,
        networkName: network.name,
      };

      // Setup event listeners
      this.setupEventListeners();

      return this.walletState;
    } catch (error: any) {
      console.error('Failed to connect wallet:', error);
      throw new Error(error.message || 'Failed to connect to MetaMask');
    }
  }

  /**
   * Switch to Hardhat local network
   */
  async switchToHardhatNetwork(): Promise<void> {
    if (!this.isMetaMaskInstalled()) {
      throw new Error('MetaMask is not installed');
    }

    try {
      // Try to switch to Hardhat network
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: HARDHAT_NETWORK.chainId }],
      });
    } catch (switchError: any) {
      // Network not added, add it
      if (switchError.code === 4902) {
        try {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [HARDHAT_NETWORK],
          });
        } catch (addError) {
          throw new Error('Failed to add Hardhat network to MetaMask');
        }
      } else {
        throw new Error('Failed to switch to Hardhat network');
      }
    }
  }

  /**
   * Disconnect wallet
   */
  disconnectWallet(): void {
    this.provider = null;
    this.signer = null;
    this.walletState = {
      connected: false,
      address: null,
      balance: null,
      chainId: null,
      networkName: null,
    };
  }

  /**
   * Get current wallet state
   */
  getWalletState(): WalletState {
    return { ...this.walletState };
  }

  /**
   * Get provider
   */
  getProvider(): BrowserProvider | null {
    return this.provider;
  }

  /**
   * Get signer
   */
  getSigner(): JsonRpcSigner | null {
    return this.signer;
  }

  /**
   * Get current address
   */
  getAddress(): string | null {
    return this.walletState.address;
  }

  /**
   * Get balance
   */
  async getBalance(address?: string): Promise<string> {
    if (!this.provider) {
      throw new Error('Provider not initialized');
    }

    const addr = address || this.walletState.address;
    if (!addr) {
      throw new Error('No address provided');
    }

    const balance = await this.provider.getBalance(addr);
    return formatEther(balance);
  }

  /**
   * Send transaction
   */
  async sendTransaction(to: string, value: string): Promise<any> {
    if (!this.signer) {
      throw new Error('Signer not initialized');
    }

    const tx = await this.signer.sendTransaction({
      to,
      value: parseEther(value),
    });

    return tx;
  }

  /**
   * Sign message
   */
  async signMessage(message: string): Promise<string> {
    if (!this.signer) {
      throw new Error('Signer not initialized');
    }

    return await this.signer.signMessage(message);
  }

  /**
   * Get contract instance
   */
  getContract(address: string, abi: any): Contract {
    if (!this.provider) {
      throw new Error('Provider not initialized');
    }

    return new Contract(address, abi, this.signer || this.provider);
  }

  /**
   * Setup event listeners for wallet changes
   */
  private setupEventListeners(): void {
    if (!window.ethereum) return;

    // Account changed
    window.ethereum.on('accountsChanged', async (accounts: string[]) => {
      if (accounts.length === 0) {
        this.disconnectWallet();
      } else {
        // Reconnect with new account
        await this.connectWallet();
      }
    });

    // Chain changed
    window.ethereum.on('chainChanged', () => {
      // Reload page on chain change
      window.location.reload();
    });

    // Disconnect
    window.ethereum.on('disconnect', () => {
      this.disconnectWallet();
    });
  }

  /**
   * Format address for display
   */
  formatAddress(address: string): string {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  }

  /**
   * Check if connected to correct network
   */
  async isCorrectNetwork(): Promise<boolean> {
    if (!this.provider) return false;
    const network = await this.provider.getNetwork();
    return network.chainId === BigInt(31337);
  }

  /**
   * Get network name
   */
  getNetworkName(chainId: number): string {
    const networks: { [key: number]: string } = {
      1: 'Ethereum Mainnet',
      3: 'Ropsten Testnet',
      4: 'Rinkeby Testnet',
      5: 'Goerli Testnet',
      137: 'Polygon Mainnet',
      80001: 'Mumbai Testnet',
      31337: 'Hardhat Local',
    };
    return networks[chainId] || 'Unknown Network';
  }
}

// Export singleton instance
export const web3Service = new Web3Service();

// Type declarations for window.ethereum
declare global {
  interface Window {
    ethereum?: any;
  }
}
