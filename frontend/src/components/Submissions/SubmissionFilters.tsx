import React from 'react';

interface SubmissionFiltersProps {
  currentStatus: string;
  onStatusChange: (status: string) => void;
}

const SubmissionFilters: React.FC<SubmissionFiltersProps> = ({
  currentStatus,
  onStatusChange,
}) => {
  const statusOptions = [
    { value: '', label: 'All Submissions' },
    { value: 'COMPLETED', label: 'Completed' },
    { value: 'PROCESSING', label: 'Processing' },
    { value: 'PENDING', label: 'Pending' },
    { value: 'REVIEW', label: 'Needs Review' },
    { value: 'FAILED', label: 'Failed' },
  ];

  const verificationStatusOptions = [
    { value: '', label: 'All Results' },
    { value: 'PASS', label: 'Passed Verification' },
    { value: 'FAIL', label: 'Failed Verification' },
    { value: 'REVIEW', label: 'Under Review' },
  ];

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-4">Filter Submissions</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Status Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Processing Status
          </label>
          <select
            value={currentStatus}
            onChange={(e) => onStatusChange(e.target.value)}
            className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          >
            {statusOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Date Range Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Date Range
          </label>
          <select className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
            <option value="">All Time</option>
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="quarter">This Quarter</option>
          </select>
        </div>

        {/* Verification Status Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Verification Result
          </label>
          <select className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
            {verificationStatusOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Quick Filters */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-sm font-medium text-gray-700 mb-2">Quick Filters</p>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => onStatusChange('REVIEW')}
            className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 hover:bg-yellow-200"
          >
            Needs Attention
          </button>
          <button
            onClick={() => onStatusChange('FAILED')}
            className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 hover:bg-red-200"
          >
            Failed Submissions
          </button>
          <button
            onClick={() => onStatusChange('PROCESSING')}
            className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 hover:bg-blue-200"
          >
            Currently Processing
          </button>
          <button
            onClick={() => onStatusChange('COMPLETED')}
            className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 hover:bg-green-200"
          >
            Successfully Verified
          </button>
        </div>
      </div>

      {/* Clear Filters */}
      {currentStatus && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <button
            onClick={() => onStatusChange('')}
            className="text-sm text-blue-600 hover:text-blue-500 font-medium"
          >
            Clear all filters
          </button>
        </div>
      )}
    </div>
  );
};

export default SubmissionFilters;