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
} from 'recharts';

interface ProfileAnalyticsProps {
  analytics: {
    consistencyTrend: Array<{ date: string; score: number }>;
    featureEvolution: Array<{ date: string; features: any }>;
    comparisonMetrics: {
      peerAverage: number;
      institutionalAverage: number;
      percentile: number;
    };
  };
}

const ProfileAnalytics: React.FC<ProfileAnalyticsProps> = ({ analytics }) => {
  const { consistencyTrend, featureEvolution, comparisonMetrics } = analytics;

  // Prepare comparison data
  const comparisonData = [
    {
      name: 'Your Score',
      value: consistencyTrend[consistencyTrend.length - 1]?.score || 0,
      color: '#3B82F6',
    },
    {
      name: 'Peer Average',
      value: comparisonMetrics.peerAverage,
      color: '#10B981',
    },
    {
      name: 'Institutional Average',
      value: comparisonMetrics.institutionalAverage,
      color: '#F59E0B',
    },
  ];

  // Prepare feature evolution data (simplified)
  const evolutionData = featureEvolution.map((item) => ({
    date: new Date(item.date).toLocaleDateString(),
    lexicalRichness: item.features.lexicalRichness?.ttr * 100 || 0,
    sentenceComplexity: item.features.syntacticComplexity?.avgSentenceLength || 0,
    vocabularyDiversity: item.features.lexicalRichness?.vocdD || 0,
  }));

  return (
    <div className="space-y-8">
      {/* Consistency Trend */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Authorship Consistency Trend</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={consistencyTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Consistency Score']}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-sm text-gray-600">
          <p>
            This chart shows how consistent your writing style has been over time. 
            Higher scores indicate more consistent authorship patterns.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Feature Evolution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Writing Style Evolution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={evolutionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="lexicalRichness"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  name="Lexical Richness"
                />
                <Line
                  type="monotone"
                  dataKey="sentenceComplexity"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Sentence Complexity"
                />
                <Line
                  type="monotone"
                  dataKey="vocabularyDiversity"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  name="Vocabulary Diversity"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Comparison Metrics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Performance Comparison</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" />
                <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                <Bar dataKey="value" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-600">Your Percentile Rank:</span>
              <span className="font-medium text-gray-900">
                {comparisonMetrics.percentile.toFixed(0)}th percentile
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-3">Consistency Score</h4>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">
              {consistencyTrend[consistencyTrend.length - 1]?.score.toFixed(1) || 0}%
            </div>
            <p className="text-sm text-gray-600 mt-1">Current consistency rating</p>
          </div>
          <div className="mt-4 text-xs text-gray-500">
            <p>
              Measures how consistently your writing style matches your established profile.
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-3">Peer Comparison</h4>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">
              {comparisonMetrics.percentile.toFixed(0)}th
            </div>
            <p className="text-sm text-gray-600 mt-1">Percentile rank</p>
          </div>
          <div className="mt-4 text-xs text-gray-500">
            <p>
              Your ranking compared to other students in similar academic programs.
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h4 className="text-md font-medium text-gray-900 mb-3">Improvement Trend</h4>
          <div className="text-center">
            {(() => {
              const recent = consistencyTrend.slice(-3);
              const trend = recent.length > 1 
                ? recent[recent.length - 1].score - recent[0].score 
                : 0;
              return (
                <>
                  <div className={`text-3xl font-bold ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {trend >= 0 ? '+' : ''}{trend.toFixed(1)}%
                  </div>
                  <p className="text-sm text-gray-600 mt-1">Recent change</p>
                </>
              );
            })()}
          </div>
          <div className="mt-4 text-xs text-gray-500">
            <p>
              Change in consistency score over the last few submissions.
            </p>
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-3">Analytics Insights</h3>
        <div className="space-y-2 text-sm text-blue-800">
          {comparisonMetrics.percentile >= 75 && (
            <p>• Your writing consistency is in the top 25% of students - excellent work!</p>
          )}
          {comparisonMetrics.percentile < 50 && (
            <p>• Consider adding more writing samples to improve your profile accuracy.</p>
          )}
          {consistencyTrend.length > 1 && (
            <p>
              • Your consistency has {
                consistencyTrend[consistencyTrend.length - 1].score > 
                consistencyTrend[consistencyTrend.length - 2].score 
                  ? 'improved' : 'decreased'
              } in your most recent submission.
            </p>
          )}
          <p>• Regular submissions help maintain and improve your authorship profile.</p>
        </div>
      </div>
    </div>
  );
};

export default ProfileAnalytics;