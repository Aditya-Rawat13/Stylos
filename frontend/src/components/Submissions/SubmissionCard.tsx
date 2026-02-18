import React from 'react';
import { Submission } from '../../services/submissionService';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  EyeIcon,
  ArrowPathIcon,
  TrashIcon,
  CubeIcon,
} from '@heroicons/react/24/outline';

interface SubmissionCardProps {
  submission: Submission;
  onRetry: (submissionId: string) => void;
  onDelete: (submissionId: string) => void;
  onViewDetails: (submissionId: string) => void;
  isRetrying: boolean;
  isDeleting: boolean;
}

const SubmissionCard: React.FC<SubmissionCardProps> = ({
  submission,
  onRetry,
  onDelete,
  onViewDetails,
  isRetrying,
  isDeleting,
}) => {
  const getStatusIcon = () => {
    const status = submission.status.toLowerCase();
    if (status === 'completed' || status === 'verified' || status === 'approved') {
      return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
    } else if (status === 'processing' || status === 'pending' || status === 'uploaded') {
      return <ClockIcon className="h-5 w-5 text-yellow-500 animate-pulse" />;
    } else if (status === 'failed' || status === 'review' || status === 'flagged' || status === 'rejected') {
      return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
    } else {
      return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    const status = submission.status.toLowerCase();
    if (status === 'completed' || status === 'verified' || status === 'approved') {
      return 'text-green-600 bg-green-100';
    } else if (status === 'processing' || status === 'pending' || status === 'uploaded') {
      return 'text-yellow-600 bg-yellow-100';
    } else if (status === 'failed' || status === 'review' || status === 'flagged' || status === 'rejected') {
      return 'text-red-600 bg-red-100';
    } else {
      return 'text-gray-600 bg-gray-100';
    }
  };

  const getOverallStatusColor = () => {
    if (!submission.verificationResult) return 'text-gray-600';
    
    switch (submission.verificationResult.overallStatus) {
      case 'PASS':
        return 'text-green-600';
      case 'FAIL':
        return 'text-red-600';
      case 'REVIEW':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
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

  const truncateContent = (content: string, maxLength: number = 150) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {/* Title and Status */}
          <div className="flex items-center space-x-3 mb-2">
            <h3 className="text-lg font-medium text-gray-900 truncate">
              {submission.title}
            </h3>
            {getStatusIcon()}
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor()}`}
            >
              {submission.status}
            </span>
          </div>

          {/* Content Preview */}
          <p className="text-sm text-gray-600 mb-4">
            {truncateContent(submission.content)}
          </p>

          {/* Verification Results */}
          {submission.verificationResult && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Authorship</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round(submission.verificationResult.authorshipScore * 100)}%
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">AI Probability</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round(submission.verificationResult.aiProbability * 100)}%
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Overall Status</span>
                <span className={`text-sm font-medium ${getOverallStatusColor()}`}>
                  {submission.verificationResult.overallStatus}
                </span>
              </div>
            </div>
          )}

          {/* Duplicate Matches */}
          {submission.verificationResult?.duplicateMatches && 
           submission.verificationResult.duplicateMatches.length > 0 && (
            <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">
                <ExclamationTriangleIcon className="inline h-4 w-4 mr-1" />
                {submission.verificationResult.duplicateMatches.length} potential duplicate(s) found
              </p>
            </div>
          )}

          {/* Metadata */}
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>Submitted: {formatDate(submission.submittedAt)}</span>
            {submission.blockchainTxHash && (
              <span className="flex items-center">
                <CubeIcon className="h-4 w-4 mr-1" />
                Blockchain verified
              </span>
            )}
            {submission.verificationResult?.processingTime && (
              <span>
                Processed in {Math.round(submission.verificationResult.processingTime / 1000)}s
              </span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2 ml-4">
          <button
            onClick={() => onViewDetails(submission.id)}
            className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <EyeIcon className="mr-2 h-4 w-4" />
            Details
          </button>

          {(submission.status === 'FAILED' || submission.status === 'REVIEW') && (
            <button
              onClick={() => onRetry(submission.id)}
              disabled={isRetrying}
              className="inline-flex items-center px-3 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isRetrying ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              ) : (
                <ArrowPathIcon className="mr-2 h-4 w-4" />
              )}
              Retry
            </button>
          )}

          <button
            onClick={() => onDelete(submission.id)}
            disabled={isDeleting}
            className="inline-flex items-center px-3 py-2 border border-red-300 rounded-md shadow-sm text-sm font-medium text-red-700 bg-white hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDeleting ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-600 mr-2"></div>
            ) : (
              <TrashIcon className="mr-2 h-4 w-4" />
            )}
            Delete
          </button>
        </div>
      </div>

      {/* Processing Indicator */}
      {(submission.status === 'PROCESSING' || submission.status === 'PENDING') && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">
              {submission.status === 'PENDING' ? 'Queued for processing...' : 'Processing...'}
            </span>
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span className="text-blue-600">In progress</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SubmissionCard;