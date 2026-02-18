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
  AreaChart,
  Area,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
} from 'recharts';

interface TrendAnalysisProps {
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

const TrendAnalysis: React.FC<TrendAnalysisProps> = ({ metrics, timeRange }) => {
  // Prepare data for trend analysis
  const submissionTrendsData = metrics.submissionTrends.map(item => ({
    ...item,
    date: new Date(item.date).toLocaleDateString(),
    failRate: 100 - item.passRate,
  }));

  const aiTrendsData = metrics.aiUsageStats.map(item => ({
    ...item,
    date: new Date(item.date).toLocaleDateString(),
    humanProbability: 100 - item.aiProbability,
  }));

  // Calculate trend indicators
  const calculateTrend = (data: Array<{ count?: number; passRate?: number; aiProbability?: number }>) => {
    if (data.length < 2) return { direction: 'stable', percentage: 0 };
    
    const recent = data.slice(-3);
    const older = data.slice(0, 3);
    
    const recentAvg = recent.reduce((sum, item) => sum + (item.count || item.passRate || item.aiProbability || 0), 0) / recent.length;
    const olderAvg = older.reduce((sum, item) => sum + (item.count || item.passRate || item.aiProbability || 0), 0) / older.length;
    
    const change = ((recentAvg - olderAvg) / olderAvg) * 100;
    
    return {
      direction: change > 5 ? 'increasing' : change < -5 ? 'decreasing' : 'stable',
      percentage: Math.abs(change),
    };
  };

  const submissionTrend = calculateTrend(metrics.submissionTrends);
  const passRateTrend = calculateTrend(metrics.submissionTrends.map(s => ({ passRate: s.passRate })));
  const aiTrend = calculateTrend(metrics.aiUsageStats.map(a => ({ aiProbability: a.aiProbability })));

  // Prepare outlier analysis data
  const outliersByType = metrics.outlierDetections.reduce((acc, outlier) => {
    acc[outlier.anomalyType] = (acc[outlier.anomalyType] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const outlierTypeData = Object.entries(outliersByType).map(([type, count]) => ({
    type,
    count,
  }));

  const outlierSeverityData = ['low', 'medium', 'high'].map(severity => ({
    severity,
    count: metrics.outlierDetections.filter(o => o.severity === severity).length,
  }));

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'increasing':
        return '↗️';
      case 'decreasing':
        return '↘️';
      default:
        return '➡️';
    }
  };

  const getTrendColor = (direction: string, isPositive: boolean = true) => {
    if (direction === 'stable') return 'text-gray-600';
    const isGood = (direction === 'increasing' && isPositive) || (direction === 'decreasing' && !isPositive);
    return isGood ? 'text-green-600' : 'text-red-600';
  };

  return (
    <div className="space-y-8">
      {/* Trend Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Submission Volume</p>
              <p className={`text-lg font-semibold ${getTrendColor(submissionTrend.direction, true)}`}>
                {getTrendIcon(submissionTrend.direction)} {submissionTrend.direction}
              </p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900">
                {metrics.submissionTrends[metrics.submissionTrends.length - 1]?.count || 0}
              </p>
              <p className="text-sm text-gray-500">
                {submissionTrend.percentage.toFixed(1)}% change
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Pass Rate</p>
              <p className={`text-lg font-semibold ${getTrendColor(passRateTrend.direction, true)}`}>
                {getTrendIcon(passRateTrend.direction)} {passRateTrend.direction}
              </p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900">
                {Math.round(metrics.submissionTrends[metrics.submissionTrends.length - 1]?.passRate || 0)}%
              </p>
              <p className="text-sm text-gray-500">
                {passRateTrend.percentage.toFixed(1)}% change
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">AI Detection</p>
              <p className={`text-lg font-semibold ${getTrendColor(aiTrend.direction, false)}`}>
                {getTrendIcon(aiTrend.direction)} {aiTrend.direction}
              </p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900">
                {Math.round(metrics.aiUsageStats[metrics.aiUsageStats.length - 1]?.aiProbability || 0)}%
              </p>
              <p className="text-sm text-gray-500">
                {aiTrend.percentage.toFixed(1)}% change
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Trend Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Submission Volume and Pass Rate Trends */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Submission Volume & Pass Rate Trends</h3>
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
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="failRate"
                  stroke="#EF4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Fail Rate (%)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* AI Detection Trends */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">AI Detection Trends</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={aiTrendsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="aiProbability"
                  stroke="#EF4444"
                  strokeWidth={3}
                  name="AI Probability (%)"
                />
                <Line
                  type="monotone"
                  dataKey="humanProbability"
                  stroke="#10B981"
                  strokeWidth={3}
                  name="Human Probability (%)"
                />
                <Line
                  type="monotone"
                  dataKey="flagged"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  name="Flagged Count"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Outlier Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Outlier Types */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Outlier Detection by Type</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={outlierTypeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8B5CF6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Outlier Severity */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Outlier Severity Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={outlierSeverityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count">
                  {outlierSeverityData.map((entry: any, index: number) => {
                    const colors: Record<string, string> = {
                      'high': '#EF4444',
                      'medium': '#F59E0B',
                      'low': '#3B82F6'
                    };
                    return <Cell key={`cell-${index}`} fill={colors[entry.severity] || '#6B7280'} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Predictive Insights */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-purple-900 mb-4">Predictive Insights & Recommendations</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-md font-medium text-purple-900 mb-2">Trend Predictions</h4>
            <div className="space-y-2 text-sm text-purple-800">
              {submissionTrend.direction === 'increasing' && (
                <p>• Submission volume is trending upward - consider scaling verification resources</p>
              )}
              {passRateTrend.direction === 'decreasing' && (
                <p>• Pass rates are declining - investigate potential causes and implement interventions</p>
              )}
              {aiTrend.direction === 'increasing' && (
                <p>• AI usage appears to be increasing - enhance detection algorithms and education</p>
              )}
              {metrics.outlierDetections.filter(o => o.severity === 'high').length > 0 && (
                <p>• High-severity outliers detected - immediate review recommended</p>
              )}
            </div>
          </div>
          
          <div>
            <h4 className="text-md font-medium text-purple-900 mb-2">Recommended Actions</h4>
            <div className="space-y-2 text-sm text-purple-800">
              <p>• Monitor submission patterns for seasonal variations</p>
              <p>• Implement proactive student education on academic integrity</p>
              <p>• Consider adjusting verification thresholds based on trends</p>
              <p>• Schedule regular reviews of flagged submissions</p>
              {metrics.duplicatePatterns.length > 0 && (
                <p>• Investigate duplicate submission patterns for systematic issues</p>
              )}
            </div>
          </div>
        </div>

        {/* Key Metrics Summary */}
        <div className="mt-6 pt-6 border-t border-purple-200">
          <h4 className="text-md font-medium text-purple-900 mb-3">Key Performance Indicators</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(metrics.submissionTrends.reduce((sum, s) => sum + s.passRate, 0) / metrics.submissionTrends.length)}%
              </div>
              <div className="text-xs text-purple-700">Avg Pass Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(metrics.aiUsageStats.reduce((sum, a) => sum + a.aiProbability, 0) / metrics.aiUsageStats.length)}%
              </div>
              <div className="text-xs text-purple-700">Avg AI Detection</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {metrics.outlierDetections.length}
              </div>
              <div className="text-xs text-purple-700">Total Outliers</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {metrics.duplicatePatterns.reduce((sum, d) => sum + d.count, 0)}
              </div>
              <div className="text-xs text-purple-700">Duplicate Cases</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrendAnalysis;