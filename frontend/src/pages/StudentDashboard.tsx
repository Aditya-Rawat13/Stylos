import React, { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { submissionService } from '../services/submissionService';
import { profileService } from '../services/profileService';
import { blockchainService } from '../services/blockchainService';
import {
  CloudArrowUpIcon,
  DocumentTextIcon,
  CubeIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';

const StudentDashboard: React.FC = () => {
  const { user } = useAuth();
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');

  const { data: recentSubmissions, isLoading: submissionsLoading } = useQuery(
    'recent-submissions',
    () => submissionService.getSubmissions(1, 5),
    { 
      refetchInterval: 30000,
      retry: 1,
      onError: (error) => {
        console.error('Failed to fetch submissions:', error);
      }
    }
  );

  const { data: writingProfile, isLoading: profileLoading } = useQuery(
    'writing-profile',
    () => profileService.getWritingProfile(),
    { 
      retry: false,
      onError: (error) => {
        console.error('Failed to fetch writing profile:', error);
      }
    }
  );

  const { data: portfolio, isLoading: portfolioLoading } = useQuery(
    'blockchain-portfolio',
    () => blockchainService.getPortfolio(),
    { 
      refetchInterval: 60000,
      retry: 1,
      onError: (error) => {
        console.error('Failed to fetch blockchain portfolio:', error);
      }
    }
  );

  const { data: analytics } = useQuery(
    ['profile-analytics', timeRange],
    () => profileService.getProfileAnalytics(timeRange),
    { enabled: !!writingProfile }
  );

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'PROCESSING':
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
      case 'FAILED':
      case 'REVIEW':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'text-green-600 bg-green-100';
      case 'PROCESSING':
        return 'text-yellow-600 bg-yellow-100';
      case 'FAILED':
      case 'REVIEW':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Welcome back, {user?.name}</h1>
        <p className="text-gray-600 mt-1">
          Track your writing authenticity and academic integrity
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <DocumentTextIcon className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Total Submissions</p>
              <p className="text-2xl font-semibold text-gray-900">
                {recentSubmissions?.total || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Profile Confidence</p>
              <p className="text-2xl font-semibold text-gray-900">
                {writingProfile ? `${Math.round(writingProfile.confidenceScore * 100)}%` : 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CubeIcon className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Blockchain Tokens</p>
              <p className="text-2xl font-semibold text-gray-900">
                {portfolio?.totalTokens || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CheckCircleIcon className="h-8 w-8 text-emerald-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Verified Essays</p>
              <p className="text-2xl font-semibold text-gray-900">
                {portfolio?.totalVerifiedSubmissions || 0}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Submissions */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-medium text-gray-900">Recent Submissions</h2>
                <Link
                  to="/submissions"
                  className="text-sm text-blue-600 hover:text-blue-500 font-medium"
                >
                  View all
                </Link>
              </div>
            </div>
            <div className="p-6">
              {submissionsLoading ? (
                <div className="space-y-4">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="animate-pulse">
                      <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                      <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                    </div>
                  ))}
                </div>
              ) : recentSubmissions?.submissions && recentSubmissions.submissions.length > 0 ? (
                <div className="space-y-4">
                  {recentSubmissions.submissions.map((submission) => (
                    <div
                      key={submission.id}
                      className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900">{submission.title}</h3>
                        <div className="flex items-center mt-1 space-x-4 text-sm text-gray-500">
                          <span>
                            {new Date(submission.submittedAt).toLocaleDateString()}
                          </span>
                          {submission.verificationResult && (
                            <span>
                              Authorship: {Math.round(submission.verificationResult.authorshipScore)}%
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                            submission.status
                          )}`}
                        >
                          {submission.status}
                        </span>
                        {getStatusIcon(submission.status)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No submissions yet</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Get started by uploading your first essay.
                  </p>
                  <div className="mt-6">
                    <Link
                      to="/upload"
                      className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                    >
                      <CloudArrowUpIcon className="mr-2 h-4 w-4" />
                      Upload Essay
                    </Link>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Quick Actions & Profile Summary */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h2>
            <div className="space-y-3">
              <Link
                to="/upload"
                className="flex items-center w-full px-4 py-3 text-left text-sm font-medium text-gray-700 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <CloudArrowUpIcon className="mr-3 h-5 w-5 text-blue-600" />
                Upload New Essay
              </Link>
              <Link
                to="/profile"
                className="flex items-center w-full px-4 py-3 text-left text-sm font-medium text-gray-700 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <ChartBarIcon className="mr-3 h-5 w-5 text-green-600" />
                Update Writing Profile
              </Link>
              <Link
                to="/blockchain"
                className="flex items-center w-full px-4 py-3 text-left text-sm font-medium text-gray-700 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <CubeIcon className="mr-3 h-5 w-5 text-purple-600" />
                View Blockchain Portfolio
              </Link>
            </div>
          </div>

          {/* Writing Profile Summary */}
          {writingProfile && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Writing Profile</h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-500">Confidence Score</span>
                  <span className="text-sm font-medium text-gray-900">
                    {Math.round(writingProfile.confidenceScore * 100)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-500">Sample Count</span>
                  <span className="text-sm font-medium text-gray-900">
                    {writingProfile.sampleCount}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-500">Last Updated</span>
                  <span className="text-sm font-medium text-gray-900">
                    {new Date(writingProfile.lastUpdated).toLocaleDateString()}
                  </span>
                </div>
                {writingProfile.sampleCount < 3 && (
                  <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-sm text-yellow-800">
                      Add more writing samples to improve profile accuracy.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Recent Blockchain Activity */}
          {portfolio?.recentActivity && Array.isArray(portfolio.recentActivity) && portfolio.recentActivity.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h2>
              <div className="space-y-3">
                {portfolio.recentActivity.slice(0, 3).map((activity, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-gray-900">{activity.description}</p>
                      <p className="text-xs text-gray-500">
                        {new Date(activity.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;