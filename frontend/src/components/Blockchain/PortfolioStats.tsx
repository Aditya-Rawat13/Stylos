import React from 'react';
import { BlockchainPortfolio } from '../../services/blockchainService';
import {
  CubeIcon,
  CheckCircleIcon,
  ChartBarIcon,
  StarIcon,
} from '@heroicons/react/24/outline';

interface PortfolioStatsProps {
  portfolio: BlockchainPortfolio;
}

const PortfolioStats: React.FC<PortfolioStatsProps> = ({ portfolio }) => {
  // Provide default values if portfolio or portfolioValue is undefined
  const portfolioValue = portfolio?.portfolioValue || {
    academicCredibility: 0,
    uniquenessScore: 0,
    consistencyRating: 0
  };

  const getCredibilityColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getCredibilityBg = (score: number) => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 60) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  // Early return if portfolio is not loaded
  if (!portfolio) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow p-6 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-1/2"></div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-1/2"></div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-1/2"></div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-8 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {/* Total Tokens */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <CubeIcon className="h-8 w-8 text-blue-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-500">Total Tokens</p>
            <p className="text-2xl font-semibold text-gray-900">{portfolio.totalTokens || 0}</p>
          </div>
        </div>
        <div className="mt-4">
          <div className="flex items-center text-sm text-gray-600">
            <span>Soulbound NFTs earned</span>
          </div>
        </div>
      </div>

      {/* Verified Submissions */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <CheckCircleIcon className="h-8 w-8 text-green-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-500">Verified Essays</p>
            <p className="text-2xl font-semibold text-gray-900">{portfolio.totalVerifiedSubmissions || 0}</p>
          </div>
        </div>
        <div className="mt-4">
          <div className="flex items-center text-sm text-gray-600">
            <span>Successfully authenticated</span>
          </div>
        </div>
      </div>

      {/* Academic Credibility */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <ChartBarIcon className={`h-8 w-8 ${getCredibilityColor(portfolioValue.academicCredibility)}`} />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-500">Academic Credibility</p>
            <p className={`text-2xl font-semibold ${getCredibilityColor(portfolioValue.academicCredibility)}`}>
              {Math.round(portfolioValue.academicCredibility)}%
            </p>
          </div>
        </div>
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                portfolioValue.academicCredibility >= 80
                  ? 'bg-green-500'
                  : portfolioValue.academicCredibility >= 60
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${portfolioValue.academicCredibility}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Uniqueness Score */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <StarIcon className="h-8 w-8 text-purple-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-500">Uniqueness Score</p>
            <p className="text-2xl font-semibold text-purple-600">
              {Math.round(portfolioValue.uniquenessScore)}%
            </p>
          </div>
        </div>
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-purple-500 h-2 rounded-full"
              style={{ width: `${portfolioValue.uniquenessScore}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Portfolio Value Breakdown */}
      <div className="md:col-span-2 lg:col-span-4 bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Portfolio Value Breakdown</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Academic Credibility Details */}
          <div className={`p-4 rounded-lg ${getCredibilityBg(portfolioValue.academicCredibility)}`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">Academic Credibility</h4>
              <span className={`text-lg font-bold ${getCredibilityColor(portfolioValue.academicCredibility)}`}>
                {Math.round(portfolioValue.academicCredibility)}%
              </span>
            </div>
            <p className="text-xs text-gray-600">
              Based on authorship verification scores and consistency across submissions.
            </p>
            <div className="mt-3 space-y-1">
              <div className="flex justify-between text-xs">
                <span>Avg. Authorship Score:</span>
                <span className="font-medium">
                  {(portfolio.tokens && portfolio.tokens.length > 0)
                    ? Math.round(
                        portfolio.tokens.reduce((sum, token) => sum + (token.verificationProof?.authorshipScore || 0), 0) /
                        portfolio.tokens.length
                      )
                    : 0}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span>Consistency Rating:</span>
                <span className="font-medium">{Math.round(portfolioValue.consistencyRating)}%</span>
              </div>
            </div>
          </div>

          {/* Uniqueness Score Details */}
          <div className="p-4 bg-purple-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">Uniqueness Score</h4>
              <span className="text-lg font-bold text-purple-600">
                {Math.round(portfolioValue.uniquenessScore)}%
              </span>
            </div>
            <p className="text-xs text-gray-600">
              Measures originality and absence of duplicate content across your submissions.
            </p>
            <div className="mt-3 space-y-1">
              <div className="flex justify-between text-xs">
                <span>Original Content:</span>
                <span className="font-medium">
                  {(portfolio.tokens || []).filter(token => 
                    token.verificationProof?.duplicateStatus === 'UNIQUE'
                  ).length} / {(portfolio.tokens || []).length}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span>Avg. AI Probability:</span>
                <span className="font-medium">
                  {(portfolio.tokens && portfolio.tokens.length > 0)
                    ? Math.round(
                        portfolio.tokens.reduce((sum, token) => sum + (token.verificationProof?.aiProbability || 0), 0) /
                        portfolio.tokens.length
                      )
                    : 0}%
                </span>
              </div>
            </div>
          </div>

          {/* Consistency Rating Details */}
          <div className="p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">Consistency Rating</h4>
              <span className="text-lg font-bold text-blue-600">
                {Math.round(portfolioValue.consistencyRating)}%
              </span>
            </div>
            <p className="text-xs text-gray-600">
              Evaluates how consistent your writing style is across different submissions.
            </p>
            <div className="mt-3 space-y-1">
              <div className="flex justify-between text-xs">
                <span>Style Variance:</span>
                <span className="font-medium">Low</span>
              </div>
              <div className="flex justify-between text-xs">
                <span>Profile Matches:</span>
                <span className="font-medium">
                  {(portfolio.tokens || []).filter(token => 
                    (token.verificationProof?.authorshipScore || 0) >= 70
                  ).length} / {(portfolio.tokens || []).length}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Portfolio Insights */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Portfolio Insights</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <p className="font-medium text-gray-900">Strengths:</p>
              <ul className="mt-1 space-y-1">
                {portfolioValue.academicCredibility >= 80 && (
                  <li>• High academic credibility score</li>
                )}
                {portfolioValue.uniquenessScore >= 80 && (
                  <li>• Excellent content originality</li>
                )}
                {portfolioValue.consistencyRating >= 80 && (
                  <li>• Consistent writing style</li>
                )}
                {(portfolio.totalTokens || 0) >= 5 && (
                  <li>• Strong verification history</li>
                )}
              </ul>
            </div>
            <div>
              <p className="font-medium text-gray-900">Recommendations:</p>
              <ul className="mt-1 space-y-1">
                {(portfolio.totalTokens || 0) < 5 && (
                  <li>• Submit more essays to build credibility</li>
                )}
                {portfolioValue.academicCredibility < 70 && (
                  <li>• Focus on improving authorship scores</li>
                )}
                {portfolioValue.uniquenessScore < 70 && (
                  <li>• Ensure content originality</li>
                )}
                {portfolioValue.consistencyRating < 70 && (
                  <li>• Maintain consistent writing style</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioStats;