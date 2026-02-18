import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { api } from '../services/authService';
import AdminAnalytics from '../components/Admin/AdminAnalytics';
import VerificationReports from '../components/Admin/VerificationReports';
import TrendAnalysis from '../components/Admin/TrendAnalysis';
import BulkOperations from '../components/Admin/BulkOperations';
import {
  ChartBarIcon,
  DocumentTextIcon,
  ArrowTrendingUpIcon,
  Cog6ToothIcon,
  UsersIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

interface AdminStats {
  totalStudents: number;
  totalSubmissions: number;
  pendingReviews: number;
  flaggedSubmissions: number;
  averageAuthorshipScore: number;
  averageAIDetection: number;
  duplicateDetections: number;
  blockchainAttestations: number;
}

interface InstitutionalMetrics {
  submissionTrends: Array<{ date: string; count: number; passRate: number }>;
  aiUsageStats: Array<{ date: string; aiProbability: number; flagged: number }>;
  duplicatePatterns: Array<{ type: string; count: number; percentage: number }>;
  outlierDetections: Array<{
    studentId: string;
    studentName: string;
    anomalyType: string;
    severity: 'low' | 'medium' | 'high';
    description: string;
  }>;
}

const AdminDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'reports' | 'trends' | 'operations'>('overview');
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');

  const { data: adminStats, isLoading: statsLoading } = useQuery(
    ['admin-stats', timeRange],
    async () => {
      const response = await api.get(`/api/v1/admin/stats?range=${timeRange}`);
      return response.data as AdminStats;
    },
    { refetchInterval: 60000 }
  );

  const { data: institutionalMetrics, isLoading: metricsLoading } = useQuery(
    ['institutional-metrics', timeRange],
    async () => {
      const response = await api.get(`/api/v1/admin/metrics?range=${timeRange}`);
      return response.data as InstitutionalMetrics;
    },
    { refetchInterval: 120000 }
  );

  const { data: flaggedSubmissions } = useQuery(
    'flagged-submissions',
    async () => {
      const response = await api.get('/api/v1/admin/flagged-submissions');
      return response.data;
    },
    { refetchInterval: 30000 }
  );

  const tabs = [
    { id: 'overview', name: 'Analytics Overview', icon: ChartBarIcon },
    { id: 'reports', name: 'Verification Reports', icon: DocumentTextIcon },
    { id: 'trends', name: 'Trend Analysis', icon: ArrowTrendingUpIcon },
    { id: 'operations', name: 'Bulk Operations', icon: Cog6ToothIcon },
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-blue-600 bg-blue-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  if (statsLoading || metricsLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded"></div>
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
            <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
            <p className="text-gray-600 mt-1">
              Institutional analytics and academic integrity monitoring
            </p>
          </div>
          <div className="flex items-center space-x-3">
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
        </div>
      </div>

      {/* Key Metrics */}
      {adminStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <UsersIcon className="h-8 w-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Students</p>
                <p className="text-2xl font-semibold text-gray-900">{adminStats.totalStudents}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <DocumentTextIcon className="h-8 w-8 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Submissions</p>
                <p className="text-2xl font-semibold text-gray-900">{adminStats.totalSubmissions}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <ExclamationTriangleIcon className="h-8 w-8 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Pending Reviews</p>
                <p className="text-2xl font-semibold text-gray-900">{adminStats.pendingReviews}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <ExclamationTriangleIcon className="h-8 w-8 text-red-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Flagged Submissions</p>
                <p className="text-2xl font-semibold text-gray-900">{adminStats.flaggedSubmissions}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Alert Panel for High Priority Items */}
      {institutionalMetrics?.outlierDetections && institutionalMetrics.outlierDetections.length > 0 && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center mb-3">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-400 mr-2" />
            <h3 className="text-sm font-medium text-red-800">Outlier Detections Requiring Attention</h3>
          </div>
          <div className="space-y-2">
            {institutionalMetrics.outlierDetections
              .filter(outlier => outlier.severity === 'high')
              .slice(0, 3)
              .map((outlier, index) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span className="text-red-700">
                    {outlier.studentName}: {outlier.description}
                  </span>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(outlier.severity)}`}>
                    {outlier.severity.toUpperCase()}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

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
      {activeTab === 'overview' && adminStats && institutionalMetrics && (
        <AdminAnalytics 
          stats={adminStats} 
          metrics={institutionalMetrics}
          timeRange={timeRange}
        />
      )}

      {activeTab === 'reports' && (
        <VerificationReports 
          flaggedSubmissions={flaggedSubmissions}
          timeRange={timeRange}
        />
      )}

      {activeTab === 'trends' && institutionalMetrics && (
        <TrendAnalysis 
          metrics={institutionalMetrics}
          timeRange={timeRange}
        />
      )}

      {activeTab === 'operations' && (
        <BulkOperations />
      )}

      {/* Summary Statistics */}
      {adminStats && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Verification Quality</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Avg. Authorship Score</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round(adminStats.averageAuthorshipScore)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Avg. AI Detection</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round(adminStats.averageAIDetection)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Duplicate Detections</span>
                <span className="text-sm font-medium text-gray-900">
                  {adminStats.duplicateDetections}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">System Performance</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Blockchain Attestations</span>
                <span className="text-sm font-medium text-gray-900">
                  {adminStats.blockchainAttestations}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Success Rate</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round((adminStats.blockchainAttestations / adminStats.totalSubmissions) * 100)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Review Rate</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round((adminStats.pendingReviews / adminStats.totalSubmissions) * 100)}%
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Academic Integrity</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Flagged Rate</span>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round((adminStats.flaggedSubmissions / adminStats.totalSubmissions) * 100)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Pass Rate</span>
                <span className="text-sm font-medium text-green-600">
                  {Math.round(((adminStats.totalSubmissions - adminStats.flaggedSubmissions) / adminStats.totalSubmissions) * 100)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Active Students</span>
                <span className="text-sm font-medium text-gray-900">
                  {adminStats.totalStudents}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;