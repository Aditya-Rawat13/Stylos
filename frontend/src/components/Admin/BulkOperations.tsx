import React, { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { api } from '../../services/authService';
import { useNotifications } from '../../contexts/NotificationContext';
import {
  PlayIcon,
  StopIcon,
  ArrowPathIcon,
  DocumentArrowDownIcon,
  TrashIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

interface BulkOperation {
  id: string;
  type: 'BULK_VERIFY' | 'BULK_REVIEW' | 'BULK_DELETE' | 'BULK_EXPORT' | 'SYSTEM_MAINTENANCE';
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  progress: number;
  totalItems: number;
  processedItems: number;
  startedAt: string;
  completedAt?: string;
  createdBy: string;
  parameters: any;
  results?: any;
  error?: string;
}

const BulkOperations: React.FC = () => {
  const [selectedOperation, setSelectedOperation] = useState<string>('');
  const [operationParams, setOperationParams] = useState<any>({});
  const [showConfirmation, setShowConfirmation] = useState(false);
  const { addNotification } = useNotifications();
  const queryClient = useQueryClient();

  const { data: operations, isLoading } = useQuery(
    'bulk-operations',
    async () => {
      const response = await api.get('/api/v1/admin/bulk-operations');
      return response.data as BulkOperation[];
    },
    { refetchInterval: 5000 }
  );

  const startOperationMutation = useMutation(
    async ({ type, parameters }: { type: string; parameters: any }) => {
      const response = await api.post('/api/v1/admin/bulk-operations', {
        type,
        parameters,
      });
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('bulk-operations');
        addNotification({
          type: 'success',
          title: 'Operation started',
          message: 'Bulk operation has been initiated successfully.',
        });
        setShowConfirmation(false);
        setSelectedOperation('');
        setOperationParams({});
      },
      onError: (error: any) => {
        addNotification({
          type: 'error',
          title: 'Operation failed',
          message: error.response?.data?.message || 'Failed to start bulk operation.',
        });
      },
    }
  );

  const cancelOperationMutation = useMutation(
    async (operationId: string) => {
      const response = await api.post(`/api/v1/admin/bulk-operations/${operationId}/cancel`);
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('bulk-operations');
        addNotification({
          type: 'success',
          title: 'Operation cancelled',
          message: 'Bulk operation has been cancelled successfully.',
        });
      },
      onError: (error: any) => {
        addNotification({
          type: 'error',
          title: 'Cancellation failed',
          message: error.response?.data?.message || 'Failed to cancel operation.',
        });
      },
    }
  );

  const operationTypes = [
    {
      id: 'BULK_VERIFY',
      name: 'Bulk Verification',
      description: 'Re-run verification on selected submissions',
      icon: CheckCircleIcon,
      color: 'text-blue-600',
      params: [
        { name: 'submissionIds', type: 'textarea', label: 'Submission IDs (one per line)' },
        { name: 'forceReprocess', type: 'checkbox', label: 'Force reprocess completed submissions' },
      ],
    },
    {
      id: 'BULK_REVIEW',
      name: 'Bulk Review',
      description: 'Apply review decisions to multiple submissions',
      icon: DocumentArrowDownIcon,
      color: 'text-green-600',
      params: [
        { name: 'submissionIds', type: 'textarea', label: 'Submission IDs (one per line)' },
        { name: 'action', type: 'select', label: 'Review Action', options: ['approve', 'reject'] },
        { name: 'notes', type: 'text', label: 'Review Notes' },
      ],
    },
    {
      id: 'BULK_DELETE',
      name: 'Bulk Delete',
      description: 'Delete multiple submissions (use with caution)',
      icon: TrashIcon,
      color: 'text-red-600',
      params: [
        { name: 'submissionIds', type: 'textarea', label: 'Submission IDs (one per line)' },
        { name: 'deleteBlockchainRecords', type: 'checkbox', label: 'Also delete blockchain records' },
        { name: 'confirmationText', type: 'text', label: 'Type "DELETE" to confirm' },
      ],
    },
    {
      id: 'BULK_EXPORT',
      name: 'Bulk Export',
      description: 'Export data for multiple submissions',
      icon: DocumentArrowDownIcon,
      color: 'text-purple-600',
      params: [
        { name: 'dateRange', type: 'select', label: 'Date Range', options: ['7d', '30d', '90d', '1y', 'all'] },
        { name: 'format', type: 'select', label: 'Export Format', options: ['csv', 'json', 'pdf'] },
        { name: 'includeContent', type: 'checkbox', label: 'Include submission content' },
        { name: 'includeVerificationDetails', type: 'checkbox', label: 'Include verification details' },
      ],
    },
    {
      id: 'SYSTEM_MAINTENANCE',
      name: 'System Maintenance',
      description: 'Perform system cleanup and optimization',
      icon: ArrowPathIcon,
      color: 'text-orange-600',
      params: [
        { name: 'cleanupTempFiles', type: 'checkbox', label: 'Clean up temporary files' },
        { name: 'optimizeDatabase', type: 'checkbox', label: 'Optimize database' },
        { name: 'rebuildIndexes', type: 'checkbox', label: 'Rebuild search indexes' },
        { name: 'clearCache', type: 'checkbox', label: 'Clear application cache' },
      ],
    },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'RUNNING':
        return <PlayIcon className="h-5 w-5 text-blue-500 animate-pulse" />;
      case 'FAILED':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      case 'CANCELLED':
        return <StopIcon className="h-5 w-5 text-gray-500" />;
      default:
        return <PlayIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'text-green-600 bg-green-100';
      case 'RUNNING':
        return 'text-blue-600 bg-blue-100';
      case 'FAILED':
        return 'text-red-600 bg-red-100';
      case 'CANCELLED':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-yellow-600 bg-yellow-100';
    }
  };

  const handleStartOperation = () => {
    const operationType = operationTypes.find(op => op.id === selectedOperation);
    if (!operationType) return;

    // Validate required parameters
    const requiredParams = operationType.params.filter(p => p.type !== 'checkbox');
    const missingParams = requiredParams.filter(p => !operationParams[p.name]);
    
    if (missingParams.length > 0) {
      addNotification({
        type: 'error',
        title: 'Missing parameters',
        message: `Please fill in all required fields: ${missingParams.map(p => p.label).join(', ')}`,
      });
      return;
    }

    // Special validation for delete operation
    if (selectedOperation === 'BULK_DELETE' && operationParams.confirmationText !== 'DELETE') {
      addNotification({
        type: 'error',
        title: 'Confirmation required',
        message: 'Please type "DELETE" to confirm this destructive operation.',
      });
      return;
    }

    setShowConfirmation(true);
  };

  const confirmOperation = () => {
    startOperationMutation.mutate({
      type: selectedOperation,
      parameters: operationParams,
    });
  };

  const renderParameterInput = (param: any) => {
    switch (param.type) {
      case 'text':
        return (
          <input
            type="text"
            value={operationParams[param.name] || ''}
            onChange={(e) => setOperationParams((prev: Record<string, any>) => ({ ...prev, [param.name]: e.target.value }))}
            className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            placeholder={param.label}
          />
        );
      case 'textarea':
        return (
          <textarea
            rows={4}
            value={operationParams[param.name] || ''}
            onChange={(e) => setOperationParams((prev: Record<string, any>) => ({ ...prev, [param.name]: e.target.value }))}
            className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            placeholder={param.label}
          />
        );
      case 'select':
        return (
          <select
            value={operationParams[param.name] || ''}
            onChange={(e) => setOperationParams((prev: Record<string, any>) => ({ ...prev, [param.name]: e.target.value }))}
            className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          >
            <option value="">Select {param.label}</option>
            {param.options.map((option: string) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        );
      case 'checkbox':
        return (
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={operationParams[param.name] || false}
              onChange={(e) => setOperationParams((prev: Record<string, any>) => ({ ...prev, [param.name]: e.target.checked }))}
              className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">{param.label}</span>
          </label>
        );
      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-200 rounded w-1/3"></div>
        <div className="h-64 bg-gray-200 rounded"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* New Operation Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Start New Bulk Operation</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {operationTypes.map((opType) => (
            <div
              key={opType.id}
              onClick={() => setSelectedOperation(opType.id)}
              className={`p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                selectedOperation === opType.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center space-x-3">
                <opType.icon className={`h-6 w-6 ${opType.color}`} />
                <div>
                  <h4 className="text-sm font-medium text-gray-900">{opType.name}</h4>
                  <p className="text-xs text-gray-500">{opType.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {selectedOperation && (
          <div className="border-t border-gray-200 pt-6">
            <h4 className="text-md font-medium text-gray-900 mb-4">Operation Parameters</h4>
            <div className="space-y-4">
              {operationTypes
                .find(op => op.id === selectedOperation)
                ?.params.map((param) => (
                  <div key={param.name}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {param.label}
                    </label>
                    {renderParameterInput(param)}
                  </div>
                ))}
            </div>
            
            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => {
                  setSelectedOperation('');
                  setOperationParams({});
                }}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleStartOperation}
                disabled={startOperationMutation.isLoading}
                className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
              >
                {startOperationMutation.isLoading ? 'Starting...' : 'Start Operation'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Active Operations */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Operation History</h3>
        </div>
        
        <div className="divide-y divide-gray-200">
          {operations && operations.length > 0 ? (
            operations.map((operation) => (
              <div key={operation.id} className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(operation.status)}
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">
                        {operationTypes.find(op => op.id === operation.type)?.name || operation.type}
                      </h4>
                      <p className="text-sm text-gray-500">
                        Started by {operation.createdBy} on {new Date(operation.startedAt).toLocaleString()}
                      </p>
                      {operation.error && (
                        <p className="text-sm text-red-600 mt-1">{operation.error}</p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(operation.status)}`}>
                        {operation.status}
                      </span>
                      <p className="text-sm text-gray-500 mt-1">
                        {operation.processedItems} / {operation.totalItems} items
                      </p>
                    </div>
                    
                    {operation.status === 'RUNNING' && (
                      <button
                        onClick={() => cancelOperationMutation.mutate(operation.id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        <StopIcon className="h-5 w-5" />
                      </button>
                    )}
                  </div>
                </div>
                
                {operation.status === 'RUNNING' && (
                  <div className="mt-4">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${operation.progress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-500 mt-1">{operation.progress}% complete</p>
                  </div>
                )}
                
                {operation.results && (
                  <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                    <h5 className="text-sm font-medium text-gray-900 mb-2">Results</h5>
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                      {JSON.stringify(operation.results, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="p-6 text-center text-gray-500">
              No bulk operations have been performed yet.
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      {showConfirmation && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-yellow-100 sm:mx-0 sm:h-10 sm:w-10">
                    <ExclamationTriangleIcon className="h-6 w-6 text-yellow-600" />
                  </div>
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Confirm Bulk Operation
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to start this bulk operation? This action may affect multiple submissions and cannot be easily undone.
                      </p>
                      <div className="mt-3 p-3 bg-gray-50 rounded">
                        <p className="text-sm font-medium text-gray-900">
                          Operation: {operationTypes.find(op => op.id === selectedOperation)?.name}
                        </p>
                        <p className="text-xs text-gray-600 mt-1">
                          {JSON.stringify(operationParams, null, 2)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button
                  onClick={confirmOperation}
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-red-600 text-base font-medium text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 sm:ml-3 sm:w-auto sm:text-sm"
                >
                  Confirm
                </button>
                <button
                  onClick={() => setShowConfirmation(false)}
                  className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BulkOperations;