import React from 'react';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

interface NetworkStatusProps {
  networkStats?: {
    network?: string;
    blockHeight?: number;
    gasPrice?: string;
    networkStatus?: 'HEALTHY' | 'CONGESTED' | 'OFFLINE';
    estimatedConfirmationTime?: number;
  };
}

const NetworkStatus: React.FC<NetworkStatusProps> = ({ networkStats }) => {
  // Provide default values if networkStats is undefined or incomplete
  const safeNetworkStats = {
    network: networkStats?.network || 'Unknown',
    blockHeight: networkStats?.blockHeight || 0,
    gasPrice: networkStats?.gasPrice || '0',
    networkStatus: networkStats?.networkStatus || 'OFFLINE' as const,
    estimatedConfirmationTime: networkStats?.estimatedConfirmationTime || 0,
  };

  // Early return if networkStats is completely undefined
  if (!networkStats) {
    return (
      <div className="border rounded-lg p-4 bg-gray-50 border-gray-200">
        <div className="flex items-center space-x-3">
          <InformationCircleIcon className="h-5 w-5 text-gray-500" />
          <div>
            <h3 className="text-sm font-medium text-gray-800">Loading Network Status...</h3>
            <p className="text-sm text-gray-600">Connecting to blockchain network...</p>
          </div>
        </div>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (safeNetworkStats.networkStatus) {
      case 'HEALTHY':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'CONGESTED':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'OFFLINE':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (safeNetworkStats.networkStatus) {
      case 'HEALTHY':
        return 'bg-green-50 border-green-200';
      case 'CONGESTED':
        return 'bg-yellow-50 border-yellow-200';
      case 'OFFLINE':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const getStatusTextColor = () => {
    switch (safeNetworkStats.networkStatus) {
      case 'HEALTHY':
        return 'text-green-800';
      case 'CONGESTED':
        return 'text-yellow-800';
      case 'OFFLINE':
        return 'text-red-800';
      default:
        return 'text-gray-800';
    }
  };

  const getStatusMessage = () => {
    switch (safeNetworkStats.networkStatus) {
      case 'HEALTHY':
        return 'Network is operating normally. Transactions should confirm quickly.';
      case 'CONGESTED':
        return 'Network is experiencing high traffic. Transactions may take longer to confirm.';
      case 'OFFLINE':
        return 'Network connectivity issues detected. Please try again later.';
      default:
        return 'Network status unknown.';
    }
  };

  const formatGasPrice = (gasPrice: string) => {
    const gwei = parseFloat(gasPrice) / 1e9;
    return `${gwei.toFixed(2)} Gwei`;
  };

  return (
    <div className={`border rounded-lg p-4 ${getStatusColor()}`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-0.5">
          {getStatusIcon()}
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <h3 className={`text-sm font-medium ${getStatusTextColor()}`}>
              {safeNetworkStats.network} Network Status: {safeNetworkStats.networkStatus}
            </h3>
            <div className="flex items-center space-x-4 text-xs text-gray-600">
              <span>Block: {safeNetworkStats.blockHeight.toLocaleString()}</span>
              <span>Gas: {formatGasPrice(safeNetworkStats.gasPrice)}</span>
              <span>~{safeNetworkStats.estimatedConfirmationTime}s confirmation</span>
            </div>
          </div>
          <p className={`text-sm ${getStatusTextColor()}`}>
            {getStatusMessage()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default NetworkStatus;