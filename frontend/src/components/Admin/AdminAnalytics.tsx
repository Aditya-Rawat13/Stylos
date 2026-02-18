import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts';

interface AdminAnalyticsProps {
  stats: {
    totalStudents: number;
    totalSubmissions: number;
    pendingReviews: number;
    flaggedSubmissions: number;
    averageAuthorshipScore: number;
    averageAIDetection: number;
    duplicateDetections: number;
    blockchainAttestations: number;
  };
  metrics: {
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
  };
  timeRange: string;
}

const AdminAnalytics: React.FC<AdminAnalyticsProps> = ({ stats, metrics, timeRange }) => {
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  // Prepare submission trends data
  const submissionTrendsData = metrics.submissionTrends.map(item => ({
    ...item,
    date: new Date(item.date).toLocaleDateString(),
  }));

  // Prepare AI usage data
  const aiUsageData = metrics.aiUsageStats.map(item => ({
    ...item,
    date: new Date(item.date).toLocaleDateString(),
  }));

  // Prepare duplicate patterns for pie chart
  const duplicateData = metrics.duplicatePatterns.map(pattern => ({
    name: pattern.type,
    value: pattern.count,
    percentage: pattern.percentage,
  }));

  // Prepare outlier severity distribution
  const outlierSeverityData = [
    {
      name: 'Low',
      value: metrics.outlierDetections.filter(o => o.severity === 'low').length,
      color: '#3B82F6',
    },
    {
      name: 'Medium',
      value: metrics.outlierDetections.filter(o => o.severity === 'medium').length,
      color: '#F59E0B',
    },
    {
      name: 'High',
      value: metrics.outlierDetections.filter(o => o.severity === 'high').length,
      color: '#EF4444',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Submission Trends */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Submission Trends & Pass Rates</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={submissionTrendsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Area
                yAxisId="left"
                type="monotone"
                dataKey="count"
                stackId="1"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.3}
                name="Submissions"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="passRate"
                stroke="#10B981"
                strokeWidth={3}
                name="Pass Rate (%)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* AI Usage Statistics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">AI Detection Trends</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={aiUsageData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="aiProbability"
                  stroke="#EF4444"
                  strokeWidth={2}
                  name="Avg AI Probability (%)"
                />
                <Line
                  type="monotone"
                  dataKey="flagged"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  name="Flagged Submissions"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Duplicate Patterns */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Duplicate Detection Patterns</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={duplicateData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {duplicateData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Outlier Detection Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Outlier Severity Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Outlier Severity</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={outlierSeverityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent High-Priority Outliers */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">High-Priority Outliers</h3>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {metrics.outlierDetections
              .filter(outlier => outlier.severity === 'high')
              .slice(0, 5)
              .map((outlier, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">{outlier.studentName}</p>
                    <p className="text-sm text-gray-600">{outlier.description}</p>
                    <p className="text-xs text-gray-500">{outlier.anomalyType}</p>
                  </div>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-red-600 bg-red-100">
                    HIGH
                  </span>
                </div>
              ))}
            {metrics.outlierDetections.filter(o => o.severity === 'high').length === 0 && (
              <p className="text-sm text-gray-500 text-center py-4">No high-priority outliers detected</p>
            )}
          </div>
        </div>
      </div>

      {/* Institutional Performance Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Institutional Performance Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {Math.round((stats.blockchainAttestations / stats.totalSubmissions) * 100)}%
            </div>
            <div className="text-sm text-gray-600">Verification Success Rate</div>
          </div>
          
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {Math.round(((stats.totalSubmissions - stats.flaggedSubmissions) / stats.totalSubmissions) * 100)}%
            </div>
            <div className="text-sm text-gray-600">Academic Integrity Rate</div>
          </div>
          
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">
              {Math.round((stats.duplicateDetections / stats.totalSubmissions) * 100)}%
            </div>
            <div className="text-sm text-gray-600">Duplicate Detection Rate</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {Math.round(stats.averageAuthorshipScore)}%
            </div>
            <div className="text-sm text-gray-600">Avg. Authorship Score</div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-4">Institutional Recommendations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-md font-medium text-blue-900 mb-2">Strengths</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              {stats.averageAuthorshipScore >= 80 && (
                <li>• High average authorship verification scores</li>
              )}
              {(stats.flaggedSubmissions / stats.totalSubmissions) < 0.1 && (
                <li>• Low academic integrity violation rate</li>
              )}
              {(stats.blockchainAttestations / stats.totalSubmissions) >= 0.9 && (
                <li>• Excellent blockchain attestation success rate</li>
              )}
              {metrics.outlierDetections.filter(o => o.severity === 'high').length === 0 && (
                <li>• No high-severity outliers detected</li>
              )}
            </ul>
          </div>
          
          <div>
            <h4 className="text-md font-medium text-blue-900 mb-2">Areas for Improvement</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              {stats.averageAuthorshipScore < 70 && (
                <li>• Consider additional writing profile training for students</li>
              )}
              {(stats.flaggedSubmissions / stats.totalSubmissions) > 0.15 && (
                <li>• Implement additional academic integrity education</li>
              )}
              {stats.pendingReviews > stats.totalSubmissions * 0.1 && (
                <li>• Review queue requires attention - consider additional reviewers</li>
              )}
              {metrics.outlierDetections.filter(o => o.severity === 'high').length > 0 && (
                <li>• Address high-priority outlier detections promptly</li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminAnalytics;