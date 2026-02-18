import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { api } from '../../services/authService';
import { useNotifications } from '../../contexts/NotificationContext';
import {
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  EyeIcon,
  DocumentArrowDownIcon,
} from '@heroicons/react/24/outline';

interface FlaggedSubmission {
  id: string;
  studentId: string;
  studentName: string;
  title: string;
  submittedAt: string;
  flagReason: string;
  authorshipScore: number;
  aiProbability: number;
  duplicateMatches: number;
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
  reviewedBy?: string;
  reviewedAt?: string;
  reviewNotes?: string;
}

interface VerificationReportsProps {
  flaggedSubmissions: FlaggedSubmission[];
  timeRange: string;
}

const VerificationReports: React.FC<VerificationReportsProps> = ({
  flaggedSubmissions,
  timeRange,
}) => {
  const [selectedSubmission, setSelectedSubmission] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('');
  const [sortBy, setSortBy] = useState<'date' | 'severity' | 'student'>('date');
  const { addNotification } = useNotifications();
  const queryClient = useQueryClient();

  const { data: submissionDetails } = useQuery(
    ['submission-details', selectedSubmission],
    async () => {
      if (!selectedSubmission) return null;
      const response = await api.get(`/api/v1/admin/submissions/${selectedSubmission}/details`);
      return response.data;
    },
    { enabled: !!selectedSubmission }
  );

  const reviewMutation = useMutation(
    async ({ submissionId, action, notes }: { submissionId: string; action: 'approve' | 'reject'; notes: string }) => {
      const response = await api.post(`/api/v1/admin/submissions/${submissionId}/review`, {
        action,
        notes,
      });
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('flagged-submissions');
        addNotification({
          type: 'success',
          title: 'Review completed',
          message: 'Submission review has been processed successfully.',
        });
      },
      onError: (error: any) => {
        addNotification({
          type: 'error',
          title: 'Review failed',
          message: error.response?.data?.message || 'Failed to process review.',
        });
      },
    }
  );

  const bulkReviewMutation = useMutation(
    async ({ submissionIds, action, notes }: { submissionIds: string[]; action: 'approve' | 'reject'; notes: string }) => {
      const response = await api.post('/api/v1/admin/submissions/bulk-review', {
        submissionIds,
        action,
        notes,
      });
      return response.data;
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('flagged-submissions');
        addNotification({
          type: 'success',
          title: 'Bulk review completed',
          message: 'Selected submissions have been reviewed successfully.',
        });
      },
      onError: (error: any) => {
        addNotification({
          type: 'error',
          title: 'Bulk review failed',
          message: error.response?.data?.message || 'Failed to process bulk review.',
        });
      },
    }
  );

  const exportReportMutation = useMutation(
    async (format: 'csv' | 'pdf') => {
      const response = await api.get(`/api/v1/admin/reports/verification?format=${format}&range=${timeRange}`, {
        responseType: 'blob',
      });
      return { blob: response.data, format };
    },
    {
      onSuccess: ({ blob, format }) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `verification-report-${timeRange}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        addNotification({
          type: 'success',
          title: 'Report exported',
          message: 'Verification report has been exported successfully.',
        });
      },
      onError: (error: any) => {
        addNotification({
          type: 'error',
          title: 'Export failed',
          message: error.response?.data?.message || 'Failed to export report.',
        });
      },
    }
  );

  const [selectedSubmissions, setSelectedSubmissions] = useState<Set<string>>(new Set());
  const [bulkAction, setBulkAction] = useState<'approve' | 'reject' | ''>('');
  const [bulkNotes, setBulkNotes] = useState('');

  const handleSelectSubmission = (submissionId: string) => {
    const newSelected = new Set(selectedSubmissions);
    if (newSelected.has(submissionId)) {
      newSelected.delete(submissionId);
    } else {
      newSelected.add(submissionId);
    }
    setSelectedSubmissions(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedSubmissions.size === filteredSubmissions.length) {
      setSelectedSubmissions(new Set());
    } else {
      setSelectedSubmissions(new Set(filteredSubmissions.map(s => s.id)));
    }
  };

  const handleBulkReview = () => {
    if (bulkAction && selectedSubmissions.size > 0) {
      bulkReviewMutation.mutate({
        submissionIds: Array.from(selectedSubmissions),
        action: bulkAction,
        notes: bulkNotes,
      });
      setSelectedSubmissions(new Set());
      setBulkAction('');
      setBulkNotes('');
    }
  };

  const getSeverityColor = (submission: FlaggedSubmission) => {
    const score = submission.authorshipScore;
    const aiProb = submission.aiProbability;
    const duplicates = submission.duplicateMatches;

    if (score < 50 || aiProb > 80 || duplicates > 2) {
      return 'text-red-600 bg-red-100';
    } else if (score < 70 || aiProb > 50 || duplicates > 0) {
      return 'text-yellow-600 bg-yellow-100';
    }
    return 'text-blue-600 bg-blue-100';
  };

  const getSeverityLevel = (submission: FlaggedSubmission) => {
    const score = submission.authorshipScore;
    const aiProb = submission.aiProbability;
    const duplicates = submission.duplicateMatches;

    if (score < 50 || aiProb > 80 || duplicates > 2) return 'HIGH';
    if (score < 70 || aiProb > 50 || duplicates > 0) return 'MEDIUM';
    return 'LOW';
  };

  const filteredSubmissions = flaggedSubmissions
    .filter(submission => !filterStatus || submission.status === filterStatus)
    .sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime();
        case 'severity':
          const severityOrder = { HIGH: 3, MEDIUM: 2, LOW: 1 };
          return severityOrder[getSeverityLevel(b) as keyof typeof severityOrder] - 
                 severityOrder[getSeverityLevel(a) as keyof typeof severityOrder];
        case 'student':
          return a.studentName.localeCompare(b.studentName);
        default:
          return 0;
      }
    });

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="">All Statuses</option>
            <option value="PENDING">Pending Review</option>
            <option value="APPROVED">Approved</option>
            <option value="REJECTED">Rejected</option>
          </select>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="date">Sort by Date</option>
            <option value="severity">Sort by Severity</option>
            <option value="student">Sort by Student</option>
          </select>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => exportReportMutation.mutate('csv')}
            className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <DocumentArrowDownIcon className="mr-2 h-4 w-4" />
            Export CSV
          </button>
          <button
            onClick={() => exportReportMutation.mutate('pdf')}
            className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <DocumentArrowDownIcon className="mr-2 h-4 w-4" />
            Export PDF
          </button>
        </div>
      </div>

      {/* Bulk Actions */}
      {selectedSubmissions.size > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-blue-900">
              {selectedSubmissions.size} submission(s) selected
            </span>
            <div className="flex items-center space-x-3">
              <select
                value={bulkAction}
                onChange={(e) => setBulkAction(e.target.value as any)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm"
              >
                <option value="">Select Action</option>
                <option value="approve">Approve All</option>
                <option value="reject">Reject All</option>
              </select>
              <input
                type="text"
                placeholder="Review notes..."
                value={bulkNotes}
                onChange={(e) => setBulkNotes(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm"
              />
              <button
                onClick={handleBulkReview}
                disabled={!bulkAction || bulkReviewMutation.isLoading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Submissions Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">
              Flagged Submissions ({filteredSubmissions.length})
            </h3>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={selectedSubmissions.size === filteredSubmissions.length && filteredSubmissions.length > 0}
                onChange={handleSelectAll}
                className="mr-2"
              />
              <span className="text-sm text-gray-600">Select All</span>
            </label>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Select
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Student
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Submission
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Scores
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Severity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredSubmissions.map((submission) => (
                <tr key={submission.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <input
                      type="checkbox"
                      checked={selectedSubmissions.has(submission.id)}
                      onChange={() => handleSelectSubmission(submission.id)}
                      className="rounded"
                    />
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        {submission.studentName}
                      </div>
                      <div className="text-sm text-gray-500">
                        ID: {submission.studentId.substring(0, 8)}...
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div>
                      <div className="text-sm font-medium text-gray-900 truncate max-w-xs">
                        {submission.title}
                      </div>
                      <div className="text-sm text-gray-500">
                        {new Date(submission.submittedAt).toLocaleDateString()}
                      </div>
                      <div className="text-xs text-gray-500">
                        {submission.flagReason}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      <div>Auth: {Math.round(submission.authorshipScore)}%</div>
                      <div>AI: {Math.round(submission.aiProbability)}%</div>
                      <div>Dupes: {submission.duplicateMatches}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(submission)}`}>
                      {getSeverityLevel(submission)}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {submission.status === 'PENDING' && (
                        <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500 mr-1" />
                      )}
                      {submission.status === 'APPROVED' && (
                        <CheckCircleIcon className="h-4 w-4 text-green-500 mr-1" />
                      )}
                      {submission.status === 'REJECTED' && (
                        <XCircleIcon className="h-4 w-4 text-red-500 mr-1" />
                      )}
                      <span className="text-sm text-gray-900">{submission.status}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() => setSelectedSubmission(submission.id)}
                      className="text-blue-600 hover:text-blue-900 mr-3"
                    >
                      <EyeIcon className="h-4 w-4" />
                    </button>
                    {submission.status === 'PENDING' && (
                      <div className="flex space-x-2">
                        <button
                          onClick={() => reviewMutation.mutate({
                            submissionId: submission.id,
                            action: 'approve',
                            notes: 'Approved via quick action'
                          })}
                          className="text-green-600 hover:text-green-900"
                        >
                          <CheckCircleIcon className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => reviewMutation.mutate({
                            submissionId: submission.id,
                            action: 'reject',
                            notes: 'Rejected via quick action'
                          })}
                          className="text-red-600 hover:text-red-900"
                        >
                          <XCircleIcon className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Submission Details Modal */}
      {selectedSubmission && submissionDetails && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" onClick={() => setSelectedSubmission(null)} />
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">Submission Details</h3>
                  <button
                    onClick={() => setSelectedSubmission(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    ×
                  </button>
                </div>
                {/* Add detailed submission view here */}
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Student</label>
                      <p className="text-sm text-gray-900">{submissionDetails.studentName}</p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Submitted</label>
                      <p className="text-sm text-gray-900">{new Date(submissionDetails.submittedAt).toLocaleString()}</p>
                    </div>
                  </div>
                  {/* Add more detailed fields as needed */}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VerificationReports;