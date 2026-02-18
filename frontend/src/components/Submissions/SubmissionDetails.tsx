import React from 'react';
import { useQuery } from 'react-query';
import { submissionService } from '../../services/submissionService';
import { profileService } from '../../services/profileService';
import {
  XMarkIcon,
  DocumentTextIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  CubeIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

interface SubmissionDetailsProps {
  submissionId: string;
  onClose: () => void;
}

const SubmissionDetails: React.FC<SubmissionDetailsProps> = ({
  submissionId,
  onClose,
}) => {
  const { data: submission, isLoading } = useQuery(
    ['submission', submissionId],
    () => submissionService.getSubmission(submissionId)
  );

  const { data: verificationReport } = useQuery(
    ['verification-report', submissionId],
    () => submissionService.getVerificationReport(submissionId),
    { enabled: !!submission && submission.status === 'COMPLETED' }
  );

  const { data: profileComparison } = useQuery(
    ['profile-comparison', submissionId],
    () => profileService.getProfileComparison(submissionId),
    { enabled: !!submission && submission.status === 'COMPLETED' }
  );

  if (isLoading) {
    return (
      <div className="fixed inset-0 z-50 overflow-y-auto">
        <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
          <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
          <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
            <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
              <div className="animate-pulse space-y-4">
                <div className="h-8 bg-gray-200 rounded w-1/3"></div>
                <div className="h-64 bg-gray-200 rounded"></div>
                <div className="h-32 bg-gray-200 rounded"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!submission) {
    return null;
  }

  const getStatusIcon = () => {
    const status = submission.status.toLowerCase();
    if (status === 'completed' || status === 'verified' || status === 'approved') {
      return <CheckCircleIcon className="h-6 w-6 text-green-500" />;
    } else if (status === 'processing' || status === 'pending' || status === 'uploaded') {
      return <ClockIcon className="h-6 w-6 text-yellow-500" />;
    } else if (status === 'failed' || status === 'review' || status === 'flagged' || status === 'rejected') {
      return <ExclamationTriangleIcon className="h-6 w-6 text-red-500" />;
    } else {
      return <ClockIcon className="h-6 w-6 text-gray-500" />;
    }
  };

  const getOverallStatusColor = () => {
    if (!submission.verificationResult) return 'text-gray-600';
    
    switch (submission.verificationResult.overallStatus) {
      case 'PASS':
        return 'text-green-600 bg-green-100';
      case 'FAIL':
        return 'text-red-600 bg-red-100';
      case 'REVIEW':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  // Prepare radar chart data for profile comparison
  const radarData = profileComparison?.visualComparison.radarChart || [];

  // Prepare deviation data for bar chart
  const deviationData = profileComparison?.deviations.map(deviation => ({
    feature: deviation.feature,
    expected: deviation.expected,
    actual: deviation.actual,
    significance: deviation.significance,
  })) || [];

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" onClick={onClose} />

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-6xl sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4 max-h-screen overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                {getStatusIcon()}
                <h3 className="text-lg font-medium text-gray-900">{submission.title}</h3>
              </div>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>

            {/* Status and Metadata */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Status</span>
                  <span className="text-sm font-medium text-gray-900">{submission.status}</span>
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Submitted</span>
                  <span className="text-sm font-medium text-gray-900">
                    {new Date(submission.submittedAt).toLocaleDateString()}
                  </span>
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Content Hash</span>
                  <span className="text-sm font-medium text-gray-900 font-mono">
                    {submission.contentHash.substring(0, 12)}...
                  </span>
                </div>
              </div>
            </div>

            {/* Verification Results */}
            {submission.verificationResult && (
              <div className="mb-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">Verification Results</h4>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {Math.round(submission.verificationResult.authorshipScore * 100)}%
                      </div>
                      <div className="text-sm text-gray-600">Authorship Match</div>
                    </div>
                  </div>
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {Math.round(submission.verificationResult.aiProbability * 100)}%
                      </div>
                      <div className="text-sm text-gray-600">AI Probability</div>
                    </div>
                  </div>
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {Math.round(submission.verificationResult.confidence * 100)}%
                      </div>
                      <div className="text-sm text-gray-600">Confidence</div>
                    </div>
                  </div>
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="text-center">
                      <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getOverallStatusColor()}`}>
                        {submission.verificationResult.overallStatus}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">Overall Status</div>
                    </div>
                  </div>
                </div>

                {/* Duplicate Matches */}
                {submission.verificationResult.duplicateMatches.length > 0 && (
                  <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <h5 className="text-sm font-medium text-yellow-800 mb-2">
                      Potential Duplicates Found ({submission.verificationResult.duplicateMatches.length})
                    </h5>
                    <div className="space-y-2">
                      {submission.verificationResult.duplicateMatches.map((match, index) => (
                        <div key={index} className="flex items-center justify-between text-sm">
                          <span className="text-yellow-700">
                            {match.matchType} match with submission {match.submissionId.substring(0, 8)}...
                          </span>
                          <span className="font-medium text-yellow-800">
                            {Math.round(match.similarityScore * 100)}% similar
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Profile Comparison Visualization */}
            {profileComparison && (
              <div className="mb-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">Profile Comparison</h4>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Radar Chart */}
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <h5 className="text-sm font-medium text-gray-900 mb-3">Style Comparison</h5>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart data={radarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="feature" />
                          <PolarRadiusAxis angle={90} domain={[0, 100]} />
                          <Radar
                            name="Your Profile"
                            dataKey="profile"
                            stroke="#3B82F6"
                            fill="#3B82F6"
                            fillOpacity={0.3}
                            strokeWidth={2}
                          />
                          <Radar
                            name="This Submission"
                            dataKey="submission"
                            stroke="#EF4444"
                            fill="#EF4444"
                            fillOpacity={0.3}
                            strokeWidth={2}
                          />
                          <Legend />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Deviations */}
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <h5 className="text-sm font-medium text-gray-900 mb-3">Feature Deviations</h5>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={deviationData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="feature" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="expected" fill="#3B82F6" name="Expected" />
                          <Bar dataKey="actual" fill="#EF4444" name="Actual" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Profile Match Score */}
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-blue-900">Profile Match Score</span>
                    <span className="text-lg font-bold text-blue-900">
                      {Math.round(profileComparison.profileMatch * 100)}%
                    </span>
                  </div>
                  <div className="mt-2 w-full bg-blue-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${profileComparison.profileMatch * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            )}

            {/* Blockchain Information */}
            {submission.blockchainTxHash && (
              <div className="mb-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">Blockchain Attestation</h4>
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <CubeIcon className="h-6 w-6 text-green-600" />
                    <div>
                      <p className="text-sm font-medium text-green-900">
                        Successfully attested on blockchain
                      </p>
                      <p className="text-sm text-green-700">
                        Transaction: {submission.blockchainTxHash}
                      </p>
                      {submission.ipfsHash && (
                        <p className="text-sm text-green-700">
                          IPFS Hash: {submission.ipfsHash}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Detailed Analysis */}
            {verificationReport && (
              <div className="mb-6">
                <h4 className="text-md font-medium text-gray-900 mb-4">Detailed Analysis</h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="space-y-3">
                    {verificationReport.recommendations.map((recommendation, index) => (
                      <div key={index} className="flex items-start space-x-2">
                        <span className="text-blue-500 mt-1">•</span>
                        <span className="text-sm text-gray-700">{recommendation}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Content Preview */}
            <div className="mb-6">
              <h4 className="text-md font-medium text-gray-900 mb-4">Content Preview</h4>
              <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {submission.content.substring(0, 1000)}
                  {submission.content.length > 1000 && '...'}
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
              <button
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubmissionDetails;