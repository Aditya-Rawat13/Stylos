import React from 'react';
import { WritingProfile } from '../../services/profileService';
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
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface ProfileVisualizationProps {
  profile: WritingProfile;
}

const ProfileVisualization: React.FC<ProfileVisualizationProps> = ({ profile }) => {
  const { stylometricFeatures } = profile;

  // Prepare radar chart data for writing style overview
  const radarData = [
    {
      feature: 'Lexical Richness',
      value: Math.min(stylometricFeatures.lexicalRichness.ttr * 100, 100),
      fullMark: 100,
    },
    {
      feature: 'Sentence Complexity',
      value: Math.min(stylometricFeatures.syntacticComplexity.avgSentenceLength / 30 * 100, 100),
      fullMark: 100,
    },
    {
      feature: 'Vocabulary Diversity',
      value: Math.min(stylometricFeatures.lexicalRichness.vocdD / 100 * 100, 100),
      fullMark: 100,
    },
    {
      feature: 'Punctuation Variety',
      value: Math.min(
        (stylometricFeatures.punctuationPatterns.commaFrequency +
         stylometricFeatures.punctuationPatterns.semicolonFrequency +
         stylometricFeatures.punctuationPatterns.exclamationFrequency) * 1000,
        100
      ),
      fullMark: 100,
    },
  ];

  // Prepare bar chart data for syntactic complexity
  const syntacticData = [
    {
      name: 'Avg Sentence Length',
      value: stylometricFeatures.syntacticComplexity.avgSentenceLength,
      benchmark: 20, // Average benchmark
    },
    {
      name: 'Avg Clause Length',
      value: stylometricFeatures.syntacticComplexity.avgClauseLength,
      benchmark: 12,
    },
    {
      name: 'Subordination Ratio',
      value: stylometricFeatures.syntacticComplexity.subordinationRatio * 100,
      benchmark: 30,
    },
  ];

  // Prepare POS tag distribution data
  const posData = Object.entries(stylometricFeatures.posTagDistribution)
    .slice(0, 8) // Top 8 POS tags
    .map(([tag, frequency]) => ({
      name: tag,
      value: frequency * 100,
    }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C7C'];

  return (
    <div className="space-y-8">
      {/* Writing Style Overview */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Writing Style Overview</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="feature" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar
                name="Your Style"
                dataKey="value"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-sm text-gray-600">
          <p>
            This radar chart shows your unique writing characteristics compared to typical ranges.
            Higher values indicate stronger presence of each feature in your writing style.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Syntactic Complexity */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Syntactic Complexity</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={syntacticData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#3B82F6" name="Your Writing" />
                <Bar dataKey="benchmark" fill="#E5E7EB" name="Average" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* POS Tag Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Part-of-Speech Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={posData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {posData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Lexical Richness */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Lexical Richness</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Type-Token Ratio</span>
              <span className="text-sm font-medium text-gray-900">
                {stylometricFeatures.lexicalRichness.ttr.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">MTLD</span>
              <span className="text-sm font-medium text-gray-900">
                {stylometricFeatures.lexicalRichness.mtld.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Vocabulary Diversity</span>
              <span className="text-sm font-medium text-gray-900">
                {stylometricFeatures.lexicalRichness.vocdD.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Punctuation Patterns */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Punctuation Patterns</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Comma Frequency</span>
              <span className="text-sm font-medium text-gray-900">
                {(stylometricFeatures.punctuationPatterns.commaFrequency * 1000).toFixed(1)}/1k words
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Semicolon Frequency</span>
              <span className="text-sm font-medium text-gray-900">
                {(stylometricFeatures.punctuationPatterns.semicolonFrequency * 1000).toFixed(1)}/1k words
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Exclamation Frequency</span>
              <span className="text-sm font-medium text-gray-900">
                {(stylometricFeatures.punctuationPatterns.exclamationFrequency * 1000).toFixed(1)}/1k words
              </span>
            </div>
          </div>
        </div>

        {/* Writing Statistics */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Writing Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Total Submissions</span>
              <span className="text-sm font-medium text-gray-900">
                {profile.statistics?.totalSubmissions || 0}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Average Length</span>
              <span className="text-sm font-medium text-gray-900">
                {profile.statistics?.averageLength || 'N/A'} words
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Profile Updated</span>
              <span className="text-sm font-medium text-gray-900">
                {new Date(profile.lastUpdated).toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Function Words Analysis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Function Word Usage</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {Object.entries(stylometricFeatures.wordFrequencies.functionWords)
            .slice(0, 12)
            .map(([word, frequency]) => (
              <div key={word} className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-lg font-medium text-gray-900">"{word}"</div>
                <div className="text-sm text-gray-600">
                  {(frequency * 1000).toFixed(1)}/1k
                </div>
              </div>
            ))}
        </div>
        <div className="mt-4 text-sm text-gray-600">
          <p>
            Function words (articles, prepositions, pronouns) are strong indicators of writing style
            and are difficult to consciously modify, making them reliable for authorship verification.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ProfileVisualization;