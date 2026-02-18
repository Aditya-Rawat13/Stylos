import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { blockchainService, BlockchainRecord } from '../../services/blockchainService';
import {
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowTopRightOnSquareIcon,
  ClipboardDocumentIcon,
} from '@heroicons/react/24/outline';

const TransactionHistory: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [copiedHash, setCopiedHash] = useState<string | null>(null);
  const pageSize = 10;

  const { data: recordsData, isLoading } = useQuery(
    ['blockchain-records', currentPage],
    () => blockchainService.getBlockchainRecords(currentPage, pageSize),
    { keepPreviousData: true }
  );

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(text);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'CONFIRMED':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'PENDING':
        return <ClockIcon className="h-5 w-5 text-yellow-500 animate-pulse" />;
      case 'FAILED':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'CONFIRMED':
        return 'text-green-600 bg-green-100';
      case 'PENDING':
        return 'text-yellow-600 bg-yellow-100';
      case 'FAILED':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatHash = (hash: string) => {
    return `${hash.substring(0, 10)}...${hash.substring(hash.length - 8)}`;
  };

  const formatGasPrice = (gasPrice: string) => {
    const gwei = parseFloat(gasPrice) / 1e9;
    return `${gwei.toFixed(2)} Gwei`;
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="animate-pulse bg-white rounded-lg shadow p-6">
            <div className="flex items-center space-x-4">
              <div className="h-10 w-10 bg-gray-200 rounded-full"></div>
              <div className="flex-1 space-y-2">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
              <div className="h-8 w-20 bg-gray-200 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const records = recordsData?.records || [];
  const totalPages = recordsData ? Math.ceil(recordsData.total / pageSize) : 0;

  return (
    <div className="space-y-6">
      {/* Transaction List */}
      <div className="space-y-4">
        {records.length > 0 ? (
          records.map((record) => (
            <div key={record.id} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4 flex-1">
                  <div className="flex-shrink-0 mt-1">
                    {getStatusIcon(record.status)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-sm font-medium text-gray-900">
                        Submission Attestation
                      </h3>
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                          record.status
                        )}`}
                      >
                        {record.status}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
                      <div>
                        <p className="mb-1">
                          <span className="font-medium">Transaction Hash:</span>
                        </p>
                        <div className="flex items-center space-x-2">
                          <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono">
                            {formatHash(record.transactionHash)}
                          </code>
                          <button
                            onClick={() => copyToClipboard(record.transactionHash)}
                            className="text-gray-400 hover:text-gray-600"
                          >
                            <ClipboardDocumentIcon className="h-4 w-4" />
                          </button>
                          {copiedHash === record.transactionHash && (
                            <span className="text-xs text-green-600">Copied!</span>
                          )}
                        </div>
                      </div>

                      <div>
                        <p className="mb-1">
                          <span className="font-medium">Block Number:</span> {record.blockNumber?.toLocaleString() || 'Pending'}
                        </p>
                        <p>
                          <span className="font-medium">Timestamp:</span> {formatDate(record.timestamp)}
                        </p>
                      </div>

                      {record.status === 'CONFIRMED' && (
                        <>
                          <div>
                            <p>
                              <span className="font-medium">Gas Used:</span> {record.gasUsed?.toLocaleString() || 'N/A'}
                            </p>
                            <p>
                              <span className="font-medium">Gas Price:</span> {record.gasPrice ? formatGasPrice(record.gasPrice) : 'N/A'}
                            </p>
                          </div>

                          <div>
                            <p>
                              <span className="font-medium">Network Fee:</span> {record.networkFee || 'N/A'}
                            </p>
                            <p>
                              <span className="font-medium">Token ID:</span> {record.tokenId || 'N/A'}
                            </p>
                          </div>
                        </>
                      )}

                      {record.ipfsHash && (
                        <div className="md:col-span-2">
                          <p className="mb-1">
                            <span className="font-medium">IPFS Hash:</span>
                          </p>
                          <div className="flex items-center space-x-2">
                            <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono">
                              {formatHash(record.ipfsHash)}
                            </code>
                            <button
                              onClick={() => copyToClipboard(record.ipfsHash!)}
                              className="text-gray-400 hover:text-gray-600"
                            >
                              <ClipboardDocumentIcon className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex flex-col space-y-2 ml-4">
                  <button
                    onClick={() =>
                      window.open(`https://polygonscan.com/tx/${record.transactionHash}`, '_blank')
                    }
                    className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-xs font-medium text-gray-700 bg-white hover:bg-gray-50"
                  >
                    <ArrowTopRightOnSquareIcon className="mr-1 h-3 w-3" />
                    Explorer
                  </button>
                  
                  {record.ipfsHash && (
                    <button
                      onClick={() =>
                        window.open(`https://ipfs.io/ipfs/${record.ipfsHash}`, '_blank')
                      }
                      className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-xs font-medium text-gray-700 bg-white hover:bg-gray-50"
                    >
                      IPFS
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <ClockIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No transactions yet</h3>
            <p className="mt-1 text-sm text-gray-500">
              Submit and verify essays to see blockchain transactions here.
            </p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {(currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, recordsData?.total || 0)} of{' '}
            {recordsData?.total || 0} transactions
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            {[...Array(totalPages)].map((_, i) => {
              const page = i + 1;
              if (
                page === 1 ||
                page === totalPages ||
                (page >= currentPage - 2 && page <= currentPage + 2)
              ) {
                return (
                  <button
                    key={page}
                    onClick={() => setCurrentPage(page)}
                    className={`px-3 py-2 border rounded-md text-sm font-medium ${
                      page === currentPage
                        ? 'border-blue-500 bg-blue-50 text-blue-600'
                        : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {page}
                  </button>
                );
              } else if (page === currentPage - 3 || page === currentPage + 3) {
                return (
                  <span key={page} className="px-3 py-2 text-gray-500">
                    ...
                  </span>
                );
              }
              return null;
            })}
            <button
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TransactionHistory;