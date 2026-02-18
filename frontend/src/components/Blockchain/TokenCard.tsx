import React, { useState } from 'react';
import { SoulboundToken } from '../../services/blockchainService';
import {
  CubeIcon,
  EyeIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClipboardDocumentIcon,
} from '@heroicons/react/24/outline';

interface TokenCardProps {
  token: SoulboundToken;
}

const TokenCard: React.FC<TokenCardProps> = ({ token }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getVerificationColor = () => {
    if (token.verificationProof.authorshipScore >= 80) return 'text-green-600 bg-green-100';
    if (token.verificationProof.authorshipScore >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getAIStatusColor = () => {
    if (token.verificationProof.aiProbability <= 20) return 'text-green-600 bg-green-100';
    if (token.verificationProof.aiProbability <= 50) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            <CubeIcon className="h-8 w-8 text-blue-600" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-gray-900 truncate">
              {token.submissionTitle}
            </h3>
            <p className="text-sm text-gray-500">
              Token #{token.tokenId}
            </p>
          </div>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-gray-400 hover:text-gray-600"
        >
          <EyeIcon className="h-5 w-5" />
        </button>
      </div>

      {/* Verification Scores */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="text-lg font-semibold text-gray-900">
            {Math.round(token.verificationProof.authorshipScore)}%
          </div>
          <div className="text-xs text-gray-600">Authorship</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="text-lg font-semibold text-gray-900">
            {Math.round(token.verificationProof.aiProbability)}%
          </div>
          <div className="text-xs text-gray-600">AI Probability</div>
        </div>
      </div>

      {/* Status Badges */}
      <div className="flex flex-wrap gap-2 mb-4">
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getVerificationColor()}`}
        >
          <CheckCircleIcon className="mr-1 h-3 w-3" />
          Verified
        </span>
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getAIStatusColor()}`}
        >
          {token.verificationProof.aiProbability <= 20 ? (
            <CheckCircleIcon className="mr-1 h-3 w-3" />
          ) : (
            <ExclamationTriangleIcon className="mr-1 h-3 w-3" />
          )}
          {token.verificationProof.aiProbability <= 20 ? 'Human' : 'AI Detected'}
        </span>
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-purple-600 bg-purple-100">
          {token.verificationProof.duplicateStatus}
        </span>
      </div>

      {/* Metadata */}
      <div className="text-sm text-gray-500 mb-4">
        <p>Minted: {new Date(token.mintedAt).toLocaleDateString()}</p>
      </div>

      {/* Detailed View */}
      {showDetails && (
        <div className="border-t border-gray-200 pt-4 space-y-4">
          {/* Transaction Hash */}
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Transaction Hash
            </label>
            <div className="flex items-center space-x-2">
              <code className="text-xs bg-gray-100 px-2 py-1 rounded font-mono flex-1 truncate">
                {token.transactionHash}
              </code>
              <button
                onClick={() => copyToClipboard(token.transactionHash)}
                className="text-gray-400 hover:text-gray-600"
              >
                <ClipboardDocumentIcon className="h-4 w-4" />
              </button>
            </div>
            {copied && (
              <p className="text-xs text-green-600 mt-1">Copied to clipboard!</p>
            )}
          </div>

          {/* Attributes */}
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-2">
              Token Attributes
            </label>
            <div className="space-y-2">
              {token.ipfsMetadata.attributes.map((attr, index) => (
                <div key={index} className="flex justify-between text-xs">
                  <span className="text-gray-600">{attr.trait_type}:</span>
                  <span className="font-medium text-gray-900">{attr.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* IPFS Metadata */}
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Description
            </label>
            <p className="text-xs text-gray-600">{token.ipfsMetadata.description}</p>
          </div>

          {/* Actions */}
          <div className="flex space-x-2 pt-2">
            <button
              onClick={() => window.open(`https://polygonscan.com/tx/${token.transactionHash}`, '_blank')}
              className="flex-1 text-xs px-3 py-2 border border-gray-300 rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              View on Explorer
            </button>
            <button
              onClick={() => copyToClipboard(JSON.stringify(token, null, 2))}
              className="flex-1 text-xs px-3 py-2 border border-gray-300 rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              Copy Metadata
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TokenCard;