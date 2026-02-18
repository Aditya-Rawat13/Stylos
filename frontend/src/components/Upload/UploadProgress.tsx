import React, { useEffect, useState } from 'react';
import { submissionService, UploadProgress as ProgressType } from '../../services/submissionService';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';

interface UploadProgressProps {
  submissionId: string;
  fileName: string;
}

const UploadProgress: React.FC<UploadProgressProps> = ({ submissionId, fileName }) => {
  const [progress, setProgress] = useState<ProgressType | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);

  useEffect(() => {
    // Initial progress fetch
    const fetchProgress = async () => {
      try {
        const initialProgress = await submissionService.getUploadProgress(submissionId);
        setProgress(initialProgress);
      } catch (error) {
        console.error('Failed to fetch initial progress:', error);
      }
    };

    fetchProgress();

    // Subscribe to real-time updates
    const es = submissionService.subscribeToUpdates(submissionId, (updatedProgress) => {
      setProgress(updatedProgress);
    });

    setEventSource(es);

    return () => {
      if (es) {
        es.close();
      }
    };
  }, [submissionId]);

  if (!progress) {
    return (
      <div className="animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
        <div className="h-2 bg-gray-200 rounded"></div>
      </div>
    );
  }

  const getStageIcon = () => {
    switch (progress.stage) {
      case 'COMPLETED':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'UPLOADING':
      case 'PROCESSING':
      case 'ANALYZING':
      case 'BLOCKCHAIN':
        return <ClockIcon className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
    }
  };

  const getStageColor = () => {
    switch (progress.stage) {
      case 'COMPLETED':
        return 'bg-green-500';
      case 'UPLOADING':
      case 'PROCESSING':
      case 'ANALYZING':
      case 'BLOCKCHAIN':
        return 'bg-blue-500';
      default:
        return 'bg-red-500';
    }
  };

  const getStageDescription = () => {
    switch (progress.stage) {
      case 'UPLOADING':
        return 'Uploading file...';
      case 'PROCESSING':
        return 'Processing text content...';
      case 'ANALYZING':
        return 'Running authorship and AI analysis...';
      case 'BLOCKCHAIN':
        return 'Creating blockchain attestation...';
      case 'COMPLETED':
        return 'Verification complete!';
      default:
        return 'Processing...';
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          {getStageIcon()}
          <div>
            <p className="text-sm font-medium text-gray-900">{fileName}</p>
            <p className="text-xs text-gray-500">{getStageDescription()}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium text-gray-900">{progress.progress}%</p>
          {progress.estimatedTimeRemaining && (
            <p className="text-xs text-gray-500">
              ~{Math.round(progress.estimatedTimeRemaining / 1000)}s remaining
            </p>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${getStageColor()}`}
          style={{ width: `${progress.progress}%` }}
        ></div>
      </div>

      {/* Stage Indicators */}
      <div className="flex justify-between mt-3 text-xs">
        <div className="flex flex-col items-center">
          <div
            className={`w-3 h-3 rounded-full ${
              ['UPLOADING', 'PROCESSING', 'ANALYZING', 'BLOCKCHAIN', 'COMPLETED'].includes(
                progress.stage
              )
                ? 'bg-blue-500'
                : 'bg-gray-300'
            }`}
          ></div>
          <span className="mt-1 text-gray-500">Upload</span>
        </div>
        <div className="flex flex-col items-center">
          <div
            className={`w-3 h-3 rounded-full ${
              ['PROCESSING', 'ANALYZING', 'BLOCKCHAIN', 'COMPLETED'].includes(progress.stage)
                ? 'bg-blue-500'
                : 'bg-gray-300'
            }`}
          ></div>
          <span className="mt-1 text-gray-500">Process</span>
        </div>
        <div className="flex flex-col items-center">
          <div
            className={`w-3 h-3 rounded-full ${
              ['ANALYZING', 'BLOCKCHAIN', 'COMPLETED'].includes(progress.stage)
                ? 'bg-blue-500'
                : 'bg-gray-300'
            }`}
          ></div>
          <span className="mt-1 text-gray-500">Analyze</span>
        </div>
        <div className="flex flex-col items-center">
          <div
            className={`w-3 h-3 rounded-full ${
              ['BLOCKCHAIN', 'COMPLETED'].includes(progress.stage)
                ? 'bg-blue-500'
                : 'bg-gray-300'
            }`}
          ></div>
          <span className="mt-1 text-gray-500">Blockchain</span>
        </div>
        <div className="flex flex-col items-center">
          <div
            className={`w-3 h-3 rounded-full ${
              progress.stage === 'COMPLETED' ? 'bg-green-500' : 'bg-gray-300'
            }`}
          ></div>
          <span className="mt-1 text-gray-500">Complete</span>
        </div>
      </div>

      {/* Additional Message */}
      {progress.message && (
        <div className="mt-3 p-2 bg-gray-50 rounded text-sm text-gray-600">
          {progress.message}
        </div>
      )}
    </div>
  );
};

export default UploadProgress;