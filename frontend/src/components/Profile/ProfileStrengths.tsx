import React from 'react';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';

interface ProfileStrengthsProps {
  strengths: {
    strengths: Array<{
      category: string;
      score: number;
      description: string;
      examples: string[];
    }>;
    improvementAreas: Array<{
      category: string;
      score: number;
      suggestions: string[];
    }>;
  };
}

const ProfileStrengths: React.FC<ProfileStrengthsProps> = ({ strengths }) => {
  const { strengths: strengthsList, improvementAreas } = strengths;

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 80) return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
    if (score >= 60) return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
    return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
  };

  return (
    <div className="space-y-8">
      {/* Strengths Section */}
      <div>
        <div className="flex items-center mb-6">
          <CheckCircleIcon className="h-6 w-6 text-green-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Writing Strengths</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {strengthsList.map((strength, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">{strength.category}</h3>
                <div className="flex items-center space-x-2">
                  {getScoreIcon(strength.score)}
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getScoreColor(
                      strength.score
                    )}`}
                  >
                    {strength.score}%
                  </span>
                </div>
              </div>
              
              <p className="text-gray-600 mb-4">{strength.description}</p>
              
              {strength.examples.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Examples:</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {strength.examples.map((example, exampleIndex) => (
                      <li key={exampleIndex} className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        {example}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Improvement Areas Section */}
      <div>
        <div className="flex items-center mb-6">
          <LightBulbIcon className="h-6 w-6 text-yellow-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Areas for Improvement</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {improvementAreas.map((area, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">{area.category}</h3>
                <div className="flex items-center space-x-2">
                  {getScoreIcon(area.score)}
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getScoreColor(
                      area.score
                    )}`}
                  >
                    {area.score}%
                  </span>
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Suggestions:</h4>
                <ul className="text-sm text-gray-600 space-y-2">
                  {area.suggestions.map((suggestion, suggestionIndex) => (
                    <li key={suggestionIndex} className="flex items-start">
                      <span className="text-blue-500 mr-2">→</span>
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Overall Assessment */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-center mb-4">
          <ChartBarIcon className="h-6 w-6 text-blue-600 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">Overall Assessment</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {strengthsList.length}
            </div>
            <p className="text-sm text-gray-600">Strong Areas</p>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">
              {improvementAreas.length}
            </div>
            <p className="text-sm text-gray-600">Improvement Areas</p>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {Math.round(
                strengthsList.reduce((sum, s) => sum + s.score, 0) / strengthsList.length
              )}%
            </div>
            <p className="text-sm text-gray-600">Average Score</p>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-white rounded-lg">
          <h3 className="text-md font-medium text-gray-900 mb-2">Key Recommendations</h3>
          <div className="text-sm text-gray-600 space-y-1">
            {strengthsList.length > improvementAreas.length ? (
              <p>• Your writing shows strong consistency across multiple areas. Continue developing your unique style.</p>
            ) : (
              <p>• Focus on the improvement areas to enhance your writing profile accuracy.</p>
            )}
            <p>• Regular writing practice in your weaker areas will improve overall authorship verification.</p>
            <p>• Consider uploading more diverse writing samples to get a comprehensive profile analysis.</p>
          </div>
        </div>
      </div>

      {/* Detailed Breakdown */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Detailed Score Breakdown</h3>
        
        <div className="space-y-4">
          {[...strengthsList, ...improvementAreas]
            .sort((a, b) => b.score - a.score)
            .map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-gray-900">{item.category}</h4>
                  <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        item.score >= 80
                          ? 'bg-green-500'
                          : item.score >= 60
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${item.score}%` }}
                    ></div>
                  </div>
                </div>
                <div className="ml-4 text-right">
                  <span className="text-lg font-semibold text-gray-900">{item.score}%</span>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default ProfileStrengths;