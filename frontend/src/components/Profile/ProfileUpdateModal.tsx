import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { ProfileUpdateRequest } from '../../services/profileService';
import {
  XMarkIcon,
  CloudArrowUpIcon,
  DocumentTextIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

interface ProfileUpdateModalProps {
  onUpdate: (request: ProfileUpdateRequest) => void;
  onCancel: () => void;
  isLoading: boolean;
}

interface UploadFile extends File {
  id: string;
  sampleType: string;
  timeframe: string;
}

const ProfileUpdateModal: React.FC<ProfileUpdateModalProps> = ({
  onUpdate,
  onCancel,
  isLoading,
}) => {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [globalMetadata, setGlobalMetadata] = useState({
    defaultSampleType: 'academic',
    defaultTimeframe: 'recent',
  });

  const sampleTypes = [
    { value: 'academic', label: 'Academic Essay' },
    { value: 'creative', label: 'Creative Writing' },
    { value: 'technical', label: 'Technical Writing' },
    { value: 'personal', label: 'Personal Writing' },
    { value: 'research', label: 'Research Paper' },
    { value: 'other', label: 'Other' },
  ];

  const timeframes = [
    { value: 'recent', label: 'Recent (Last 6 months)' },
    { value: 'past_year', label: 'Past Year' },
    { value: 'older', label: 'Older than 1 year' },
    { value: 'unknown', label: 'Unknown' },
  ];

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadFile[] = acceptedFiles
      .filter(file => file && file.name && file.size !== undefined)
      .map((file) => ({
        ...file,
        id: Math.random().toString(36).substr(2, 9),
        sampleType: globalMetadata.defaultSampleType,
        timeframe: globalMetadata.defaultTimeframe,
      }));

    setFiles((prev) => [...prev, ...newFiles]);
  }, [globalMetadata]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    multiple: true,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const removeFile = (fileId: string) => {
    setFiles((prev) => prev.filter((file) => file.id !== fileId));
  };

  const updateFileMetadata = (fileId: string, field: 'sampleType' | 'timeframe', value: string) => {
    setFiles((prev) =>
      prev.map((file) =>
        file.id === fileId ? { ...file, [field]: value } : file
      )
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (files.length === 0) {
      return;
    }

    // Clean files to remove extra properties that might cause issues
    const cleanFiles = files.map(file => {
      // Create a new File object with only the necessary properties
      const cleanFile = new File([file], file.name, {
        type: file.type,
        lastModified: file.lastModified
      });
      return cleanFile;
    });

    const request: ProfileUpdateRequest = {
      samples: cleanFiles,
      metadata: {
        sampleTypes: files.map((f) => f.sampleType),
        timeframes: files.map((f) => f.timeframe),
      },
    };

    onUpdate(request);
  };

  const getFileIcon = (file: File) => {
    if (!file || !file.name) {
      return '📄';
    }
    const extension = file.name.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return '📄';
      case 'docx':
        return '📝';
      case 'txt':
        return '📃';
      default:
        return '📄';
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />

        <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-medium text-gray-900">Update Writing Profile</h3>
              <button
                onClick={onCancel}
                className="text-gray-400 hover:text-gray-600"
                disabled={isLoading}
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Global Metadata */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Default Sample Type
                  </label>
                  <select
                    value={globalMetadata.defaultSampleType}
                    onChange={(e) =>
                      setGlobalMetadata((prev) => ({
                        ...prev,
                        defaultSampleType: e.target.value,
                      }))
                    }
                    className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  >
                    {sampleTypes.map((type) => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Default Timeframe
                  </label>
                  <select
                    value={globalMetadata.defaultTimeframe}
                    onChange={(e) =>
                      setGlobalMetadata((prev) => ({
                        ...prev,
                        defaultTimeframe: e.target.value,
                      }))
                    }
                    className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  >
                    {timeframes.map((timeframe) => (
                      <option key={timeframe.value} value={timeframe.value}>
                        {timeframe.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* File Upload Area */}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-blue-400 bg-blue-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <input {...getInputProps()} />
                <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-lg font-medium text-gray-900">
                  {isDragActive ? 'Drop files here' : 'Upload writing samples'}
                </p>
                <p className="text-sm text-gray-500">
                  Drag & drop or click to select (PDF, DOCX, TXT up to 10MB each)
                </p>
              </div>

              {/* File List */}
              {files.length > 0 && (
                <div>
                  <h4 className="text-md font-medium text-gray-900 mb-4">
                    Selected Files ({files.length})
                  </h4>
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {files.filter(file => file && file.name).map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                      >
                        <div className="flex items-center space-x-3 flex-1">
                          <span className="text-2xl">{getFileIcon(file)}</span>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {file.name || 'Unknown file'}
                            </p>
                            <p className="text-sm text-gray-500">
                              {file.size ? (file.size / 1024 / 1024).toFixed(2) : '0.00'} MB
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center space-x-3">
                          <select
                            value={file.sampleType}
                            onChange={(e) =>
                              updateFileMetadata(file.id, 'sampleType', e.target.value)
                            }
                            className="text-xs border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                          >
                            {sampleTypes.map((type) => (
                              <option key={type.value} value={type.value}>
                                {type.label}
                              </option>
                            ))}
                          </select>

                          <select
                            value={file.timeframe}
                            onChange={(e) =>
                              updateFileMetadata(file.id, 'timeframe', e.target.value)
                            }
                            className="text-xs border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                          >
                            {timeframes.map((timeframe) => (
                              <option key={timeframe.value} value={timeframe.value}>
                                {timeframe.label}
                              </option>
                            ))}
                          </select>

                          <button
                            type="button"
                            onClick={() => removeFile(file.id)}
                            className="text-gray-400 hover:text-gray-600"
                          >
                            <XMarkIcon className="h-5 w-5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Info Panel */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex">
                  <InformationCircleIcon className="h-5 w-5 text-blue-400 mt-0.5" />
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-blue-800">Profile Update Guidelines</h3>
                    <div className="mt-2 text-sm text-blue-700">
                      <ul className="list-disc list-inside space-y-1">
                        <li>Upload 3-10 writing samples for best results</li>
                        <li>Include diverse writing types and timeframes</li>
                        <li>Samples should be at least 500 words each</li>
                        <li>Your existing profile will be enhanced, not replaced</li>
                        <li>Processing may take a few minutes</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Submit Buttons */}
              <div className="flex justify-end space-x-3 pt-4">
                <button
                  type="button"
                  onClick={onCancel}
                  disabled={isLoading}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading || files.length === 0}
                  className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Updating Profile...
                    </div>
                  ) : (
                    'Update Profile'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfileUpdateModal;