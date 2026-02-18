import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { profileService, ProfileUpdateRequest } from '../services/profileService';
import { useNotifications } from '../contexts/NotificationContext';
import ProfileVisualization from '../components/Profile/ProfileVisualization';
import ProfileStrengths from '../components/Profile/ProfileStrengths';
import ProfileAnalytics from '../components/Profile/ProfileAnalytics';
import ProfileUpdateModal from '../components/Profile/ProfileUpdateModal';
import {
  ChartBarIcon,
  CloudArrowUpIcon,
  DocumentArrowDownIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

const ProfilePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'analytics' | 'strengths'>('overview');
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');
  const { addNotification } = useNotifications();
  const queryClient = useQueryClient();

  const { data: profile, isLoading: profileLoading, error: profileError } = useQuery(
    'writing-profile',
    profileService.getWritingProfile,
    { retry: false }
  );

  const { data: analytics, isLoading: analyticsLoading } = useQuery(
    ['profile-analytics', timeRange],
    () => profileService.getProfileAnalytics(timeRange),
    { enabled: !!profile }
  );

  const { data: strengths, isLoading: strengthsLoading } = useQuery(
    'profile-strengths',
    profileService.getProfileStrengths,
    { enabled: !!profile }
  );

  const updateProfileMutation = useMutation(profileService.updateWritingProfile, {
    onSuccess: () => {
      queryClient.invalidateQueries('writing-profile');
      queryClient.invalidateQueries('profile-analytics');
      queryClient.invalidateQueries('profile-strengths');
      addNotification({
        type: 'success',
        title: 'Profile updated',
        message: 'Your writing profile has been updated successfully.',
      });
      setShowUpdateModal(false);
    },
    onError: (error: any) => {
      addNotification({
        type: 'error',
        title: 'Update failed',
        message: error.response?.data?.message || 'Failed to update profile.',
      });
    },
  });

  const exportProfileMutation = useMutation(profileService.exportProfile, {
    onSuccess: (blob, format) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `writing-profile.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      addNotification({
        type: 'success',
        title: 'Export successful',
        message: 'Your profile has been exported successfully.',
      });
    },
    onError: (error: any) => {
      addNotification({
        type: 'error',
        title: 'Export failed',
        message: error.response?.data?.message || 'Failed to export profile.',
      });
    },
  });

  const handleUpdateProfile = async (request: ProfileUpdateRequest) => {
    await updateProfileMutation.mutateAsync(request);
  };

  const handleExport = (format: 'json' | 'pdf' | 'csv') => {
    exportProfileMutation.mutate(format);
  };

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'analytics', name: 'Analytics', icon: ChartBarIcon },
    { id: 'strengths', name: 'Strengths', icon: ChartBarIcon },
  ];

  if (profileLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="h-64 bg-gray-200 rounded"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (profileError || !profile) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="text-center py-12">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No Writing Profile</h3>
          <p className="mt-1 text-sm text-gray-500">
            You need to create a writing profile by uploading sample essays.
          </p>
          <div className="mt-6">
            <button
              onClick={() => setShowUpdateModal(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
            >
              <CloudArrowUpIcon className="mr-2 h-4 w-4" />
              Create Profile
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Writing Profile</h1>
            <p className="text-gray-600 mt-1">
              Analyze your unique writing style and track improvements
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowUpdateModal(true)}
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              <CloudArrowUpIcon className="mr-2 h-4 w-4" />
              Update Profile
            </button>
            <div className="relative">
              <select
                value="export"
                onChange={(e) => {
                  if (e.target.value !== 'export') {
                    handleExport(e.target.value as 'json' | 'pdf' | 'csv');
                    e.target.value = 'export';
                  }
                }}
                className="appearance-none bg-white border border-gray-300 rounded-md px-4 py-2 pr-8 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="export">Export</option>
                <option value="pdf">Export as PDF</option>
                <option value="json">Export as JSON</option>
                <option value="csv">Export as CSV</option>
              </select>
              <DocumentArrowDownIcon className="absolute right-2 top-2.5 h-4 w-4 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </div>

      {/* Profile Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Confidence Score</p>
              <p className="text-2xl font-semibold text-gray-900">
                {Math.round(profile.confidenceScore * 100)}%
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <DocumentArrowDownIcon className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Sample Count</p>
              <p className="text-2xl font-semibold text-gray-900">{profile.sampleCount}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Avg. Length</p>
              <p className="text-2xl font-semibold text-gray-900">
                {profile.statistics?.averageLength || 'N/A'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-orange-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Consistency</p>
              <p className="text-2xl font-semibold text-gray-900">
                {profile.statistics?.improvementMetrics?.authorshipConsistency
                  ? `${Math.round(profile.statistics.improvementMetrics.authorshipConsistency * 100)}%`
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="mr-2 h-4 w-4" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <ProfileVisualization profile={profile} />
      )}

      {activeTab === 'analytics' && (
        <div>
          <div className="mb-6 flex items-center justify-between">
            <h2 className="text-lg font-medium text-gray-900">Analytics</h2>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="1y">Last year</option>
            </select>
          </div>
          {analyticsLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="h-64 bg-gray-200 rounded"></div>
              <div className="h-32 bg-gray-200 rounded"></div>
            </div>
          ) : analytics ? (
            <ProfileAnalytics analytics={analytics} />
          ) : (
            <div className="text-center py-8 text-gray-500">
              No analytics data available for the selected time range.
            </div>
          )}
        </div>
      )}

      {activeTab === 'strengths' && (
        <div>
          {strengthsLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="h-32 bg-gray-200 rounded"></div>
              <div className="h-32 bg-gray-200 rounded"></div>
            </div>
          ) : strengths ? (
            <ProfileStrengths strengths={strengths} />
          ) : (
            <div className="text-center py-8 text-gray-500">
              No strengths analysis available yet.
            </div>
          )}
        </div>
      )}

      {/* Profile Recommendations */}
      {profile.sampleCount < 5 && (
        <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex">
            <InformationCircleIcon className="h-5 w-5 text-yellow-400 mt-0.5" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">Improve Profile Accuracy</h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>
                  Your profile has {profile.sampleCount} writing samples. Adding more samples (recommended: 5+) 
                  will improve the accuracy of authorship verification and provide better insights into your writing style.
                </p>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setShowUpdateModal(true)}
                  className="text-yellow-800 bg-yellow-100 hover:bg-yellow-200 px-3 py-1 rounded text-sm font-medium"
                >
                  Add More Samples
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Update Profile Modal */}
      {showUpdateModal && (
        <ProfileUpdateModal
          onUpdate={handleUpdateProfile}
          onCancel={() => setShowUpdateModal(false)}
          isLoading={updateProfileMutation.isLoading}
        />
      )}
    </div>
  );
};

export default ProfilePage;