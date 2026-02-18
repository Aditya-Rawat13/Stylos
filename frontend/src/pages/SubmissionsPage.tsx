import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { submissionService, Submission } from '../services/submissionService';
import { useNotifications } from '../contexts/NotificationContext';
import SubmissionCard from '../components/Submissions/SubmissionCard';
import SubmissionFilters from '../components/Submissions/SubmissionFilters';
import SubmissionDetails from '../components/Submissions/SubmissionDetails';
import {
  DocumentTextIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

const SubmissionsPage: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSubmission, setSelectedSubmission] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const { addNotification } = useNotifications();
  const queryClient = useQueryClient();

  const pageSize = 10;

  const { data: submissionsData, isLoading, refetch } = useQuery(
    ['submissions', currentPage, statusFilter, searchQuery],
    () => submissionService.getSubmissions(currentPage, pageSize, statusFilter || undefined),
    {
      refetchInterval: 30000, // Refetch every 30 seconds for real-time updates
      keepPreviousData: true,
    }
  );

  const retryMutation = useMutation(submissionService.retryVerification, {
    onSuccess: () => {
      queryClient.invalidateQueries('submissions');
      addNotification({
        type: 'success',
        title: 'Retry initiated',
        message: 'Verification process has been restarted.',
      });
    },
    onError: (error: any) => {
      addNotification({
        type: 'error',
        title: 'Retry failed',
        message: error.response?.data?.message || 'Failed to retry verification.',
      });
    },
  });

  const deleteMutation = useMutation(submissionService.deleteSubmission, {
    onSuccess: () => {
      queryClient.invalidateQueries('submissions');
      addNotification({
        type: 'success',
        title: 'Submission deleted',
        message: 'The submission has been deleted successfully.',
      });
    },
    onError: (error: any) => {
      addNotification({
        type: 'error',
        title: 'Delete failed',
        message: error.response?.data?.message || 'Failed to delete submission.',
      });
    },
  });

  // Real-time updates for processing submissions
  useEffect(() => {
    if (!submissionsData?.submissions) return;

    const processingSubmissions = submissionsData.submissions.filter(
      (s) => s.status === 'PROCESSING' || s.status === 'PENDING'
    );

    const eventSources: EventSource[] = [];

    processingSubmissions.forEach((submission) => {
      const eventSource = submissionService.subscribeToUpdates(
        submission.id,
        (progress) => {
          // Update the submission in the cache
          queryClient.setQueryData(
            ['submissions', currentPage, statusFilter, searchQuery],
            (oldData: any) => {
              if (!oldData) return oldData;
              
              return {
                ...oldData,
                submissions: oldData.submissions.map((s: Submission) =>
                  s.id === submission.id
                    ? { ...s, status: progress.stage === 'COMPLETED' ? 'COMPLETED' : 'PROCESSING' }
                    : s
                ),
              };
            }
          );
        }
      );

      eventSources.push(eventSource);
    });

    return () => {
      eventSources.forEach((es) => es.close());
    };
  }, [submissionsData?.submissions, queryClient, currentPage, statusFilter, searchQuery]);

  const handleRetry = (submissionId: string) => {
    retryMutation.mutate(submissionId);
  };

  const handleDelete = (submissionId: string) => {
    if (window.confirm('Are you sure you want to delete this submission?')) {
      deleteMutation.mutate(submissionId);
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setCurrentPage(1);
  };

  const handleFilterChange = (status: string) => {
    setStatusFilter(status);
    setCurrentPage(1);
  };

  const filteredSubmissions = submissionsData?.submissions?.filter((submission) =>
    submission.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    submission.content.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const totalPages = submissionsData ? Math.ceil(submissionsData.total / pageSize) : 0;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">My Submissions</h1>
            <p className="text-gray-600 mt-1">
              Track your essay submissions and verification results
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
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              <FunnelIcon className="mr-2 h-4 w-4" />
              Filters
            </button>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="mb-6 space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search submissions..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        {/* Filters */}
        {showFilters && (
          <SubmissionFilters
            currentStatus={statusFilter}
            onStatusChange={handleFilterChange}
          />
        )}
      </div>

      {/* Statistics */}
      {submissionsData && submissionsData.submissions && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-gray-900">{submissionsData.total}</div>
            <div className="text-sm text-gray-600">Total Submissions</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-green-600">
              {submissionsData.submissions.filter((s) => s.status === 'COMPLETED' || s.status === 'verified' || s.status === 'VERIFIED' || s.status === 'approved' || s.status === 'APPROVED').length}
            </div>
            <div className="text-sm text-gray-600">Completed</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-yellow-600">
              {submissionsData.submissions.filter((s) => s.status === 'PROCESSING' || s.status === 'processing' || s.status === 'PENDING' || s.status === 'pending' || s.status === 'uploaded' || s.status === 'UPLOADED').length}
            </div>
            <div className="text-sm text-gray-600">Processing</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-2xl font-bold text-red-600">
              {submissionsData.submissions.filter((s) => s.status === 'FAILED' || s.status === 'failed' || s.status === 'REVIEW' || s.status === 'review' || s.status === 'flagged' || s.status === 'FLAGGED' || s.status === 'rejected' || s.status === 'REJECTED').length}
            </div>
            <div className="text-sm text-gray-600">Needs Attention</div>
          </div>
        </div>
      )}

      {/* Submissions List */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse bg-white rounded-lg shadow p-6">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 mb-4"></div>
                <div className="flex space-x-4">
                  <div className="h-8 bg-gray-200 rounded w-20"></div>
                  <div className="h-8 bg-gray-200 rounded w-16"></div>
                </div>
              </div>
            ))}
          </div>
        ) : filteredSubmissions.length > 0 ? (
          filteredSubmissions.map((submission) => (
            <SubmissionCard
              key={submission.id}
              submission={submission}
              onRetry={handleRetry}
              onDelete={handleDelete}
              onViewDetails={setSelectedSubmission}
              isRetrying={retryMutation.isLoading}
              isDeleting={deleteMutation.isLoading}
            />
          ))
        ) : (
          <div className="text-center py-12 bg-white rounded-lg shadow">
            <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              {searchQuery || statusFilter ? 'No matching submissions' : 'No submissions yet'}
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              {searchQuery || statusFilter
                ? 'Try adjusting your search or filters.'
                : 'Upload your first essay to get started.'}
            </p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-8 flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {(currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, submissionsData?.total || 0)} of{' '}
            {submissionsData?.total || 0} results
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
              } else if (
                page === currentPage - 3 ||
                page === currentPage + 3
              ) {
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

      {/* Submission Details Modal */}
      {selectedSubmission && (
        <SubmissionDetails
          submissionId={selectedSubmission}
          onClose={() => setSelectedSubmission(null)}
        />
      )}
    </div>
  );
};

export default SubmissionsPage;