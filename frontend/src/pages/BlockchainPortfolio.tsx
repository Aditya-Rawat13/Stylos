import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import { blockchainService } from '../services/blockchainService';
import { useNotifications } from '../contexts/NotificationContext';
import TokenCard from '../components/Blockchain/TokenCard';
import TransactionHistory from '../components/Blockchain/TransactionHistory';
import NetworkStatus from '../components/Blockchain/NetworkStatus';
import PortfolioStats from '../components/Blockchain/PortfolioStats';
import {
  CubeIcon,
  DocumentArrowDownIcon,
  ArrowPathIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

const BlockchainPortfolio: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'tokens' | 'transactions' | 'network'>('tokens');
  const { addNotification } = useNotifications();

  const { data: portfolio, isLoading: portfolioLoading, refetch } = useQuery(
    'blockchain-portfolio',
    blockchainService.getPortfolio,
    { refetchInterval: 60000 }
  );

  const { data: networkStats } = useQuery(
    'network-stats',
    blockchainService.getNetworkStats,
    { refetchInterval: 30000 }
  );

  const exportMutation = useMutation(blockchainService.exportPortfolio, {
    onSuccess: (blob, format) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `blockchain-portfolio.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      addNotification({
        type: 'success',
        title: 'Export successful',
        message: 'Your blockchain portfolio has been exported successfully.',
      });
    },
    onError: (error: any) => {
      addNotification({
        type: 'error',
        title: 'Export failed',
        message: error.response?.data?.message || 'Failed to export portfolio.',
      });
    },
  });

  const handleExport = (format: 'json' | 'pdf') => {
    exportMutation.mutate(format);
  };

  const tabs = [
    { id: 'tokens', name: 'Soulbound Tokens', icon: CubeIcon },
    { id: 'transactions', name: 'Transaction History', icon: ArrowPathIcon },
    { id: 'network', name: 'Network Status', icon: InformationCircleIcon },
  ];

  if (portfolioLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Blockchain Portfolio</h1>
            <p className="text-gray-600 mt-1">
              Your verified essays and proof-of-authorship tokens
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => refetch()}
              className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              <ArrowPathIcon className="mr-2 h-4 w-4" />
              Refresh
            </button>
            <div className="relative">
              <select
                value="export"
                onChange={(e) => {
                  if (e.target.value !== 'export') {
                    handleExport(e.target.value as 'json' | 'pdf');
                    e.target.value = 'export';
                  }
                }}
                className="appearance-none bg-white border border-gray-300 rounded-md px-4 py-2 pr-8 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="export">Export Portfolio</option>
                <option value="pdf">Export as PDF</option>
                <option value="json">Export as JSON</option>
              </select>
              <DocumentArrowDownIcon className="absolute right-2 top-2.5 h-4 w-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {/* Portfolio Statistics */}
      {portfolio && <PortfolioStats portfolio={portfolio} />}

      {/* Network Status Banner */}
      {networkStats && (
        <div className="mb-6">
          <NetworkStatus networkStats={networkStats} />
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="mr-2 h-4 w-4" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'tokens' && (
        <div>
          {portfolio?.tokens && portfolio.tokens.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {portfolio.tokens.map((token) => (
                <TokenCard key={token.tokenId} token={token} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-white rounded-lg shadow">
              <CubeIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No tokens yet</h3>
              <p className="mt-1 text-sm text-gray-500">
                Submit and verify essays to earn proof-of-authorship tokens.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'transactions' && (
        <TransactionHistory />
      )}

      {activeTab === 'network' && networkStats && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Network Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{networkStats.network}</div>
              <div className="text-sm text-gray-600">Network</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{(networkStats.blockHeight || 0).toLocaleString()}</div>
              <div className="text-sm text-gray-600">Block Height</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{networkStats.gasPrice || '0'} Gwei</div>
              <div className="text-sm text-gray-600">Gas Price</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{networkStats.estimatedConfirmationTime || 0}s</div>
              <div className="text-sm text-gray-600">Confirmation Time</div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center">
              <InformationCircleIcon className="h-5 w-5 text-blue-400 mr-2" />
              <div>
                <h3 className="text-sm font-medium text-blue-900">Network Status</h3>
                <p className="text-sm text-blue-700 mt-1">
                  The network is currently {(networkStats.networkStatus || 'unknown').toLowerCase()}. 
                  {networkStats.networkStatus === 'CONGESTED' && 
                    ' Transactions may take longer than usual to confirm.'}
                  {networkStats.networkStatus === 'OFFLINE' && 
                    ' Network connectivity issues detected. Please try again later.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity */}
      {portfolio?.recentActivity && portfolio.recentActivity.length > 0 && (
        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {portfolio.recentActivity.slice(0, 5).map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <div className={`w-3 h-3 rounded-full ${
                      activity.type === 'MINT' ? 'bg-green-500' :
                      activity.type === 'VERIFY' ? 'bg-blue-500' : 'bg-yellow-500'
                    }`}></div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{activity.description}</p>
                    <p className="text-sm text-gray-500">
                      {new Date(activity.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                {activity.transactionHash && (
                  <div className="text-sm text-gray-500 font-mono">
                    {activity.transactionHash.substring(0, 10)}...
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Panel */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex">
          <InformationCircleIcon className="h-5 w-5 text-blue-400 mt-0.5" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">About Blockchain Portfolio</h3>
            <div className="mt-2 text-sm text-blue-700">
              <ul className="list-disc list-inside space-y-1">
                <li>Soulbound tokens are non-transferable proof-of-authorship certificates</li>
                <li>Each verified essay receives a unique token stored on the Polygon blockchain</li>
                <li>Tokens include metadata about your writing verification and authenticity scores</li>
                <li>Your portfolio serves as permanent proof of your academic integrity</li>
                <li>All transactions are publicly verifiable on the blockchain</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BlockchainPortfolio;