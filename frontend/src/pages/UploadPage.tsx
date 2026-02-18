import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQueryClient } from 'react-query';
import { submissionService, BatchUploadRequest } from '../services/submissionService';
import { useNotifications } from '../contexts/NotificationContext';
import UploadProgress from '../components/Upload/UploadProgress';
import BatchUploadModal from '../components/Upload/BatchUploadModal';
import {
  CloudArrowUpIcon,
  DocumentTextIcon,
  XMarkIcon,
  FolderIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';

interface UploadFile extends File {
  id: string;
  progress?: number;
  status?: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  submissionId?: string;
  error?: string;
  metadata?: {
    title?: string;
    courseId?: string;
    assignmentTitle?: string;
    assignmentId?: string;
  };
}

const UploadPage: React.FC = () => {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [showBatchModal, setShowBatchModal] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState<Set<string>>(new Set());
  const [showMetadataForm, setShowMetadataForm] = useState(false);
  const [currentFileId, setCurrentFileId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState({
    title: '',
    courseId: '',
    assignmentTitle: '',
    assignmentId: '',
  });
  const { addNotification } = useNotifications();
  const queryClient = useQueryClient();

  const singleUploadMutation = useMutation(
    ({ file, metadata }: { file: File; metadata?: any }) => 
      submissionService.uploadSubmission(file, metadata),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('recent-submissions');
        queryClient.invalidateQueries('submissions');
        addNotification({
          type: 'success',
          title: 'Upload successful',
          message: 'Your essay has been uploaded and is being processed.',
        });
      },
      onError: (error: any) => {
        addNotification({
          type: 'error',
          title: 'Upload failed',
          message: error.response?.data?.detail || error.response?.data?.message || 'Failed to upload essay.',
        });
      },
    }
  );

  const batchUploadMutation = useMutation(submissionService.batchUpload, {
    onSuccess: () => {
      queryClient.invalidateQueries('recent-submissions');
      queryClient.invalidateQueries('submissions');
      addNotification({
        type: 'success',
        title: 'Batch upload successful',
        message: 'All essays have been uploaded and are being processed.',
      });
    },
    onError: (error: any) => {
      addNotification({
        type: 'error',
        title: 'Batch upload failed',
        message: error.response?.data?.detail || error.response?.data?.message || 'Failed to upload essays.',
      });
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadFile[] = acceptedFiles.map((file) => 
      Object.assign(file, {
        id: Math.random().toString(36).substr(2, 9),
        status: 'pending' as const,
      })
    );

    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

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

  const uploadSingleFile = async (file: UploadFile) => {
    if (uploadingFiles.has(file.id)) return;

    console.log('Uploading file:', file.name, 'Size:', file.size, 'Metadata:', file.metadata);
    
    setUploadingFiles((prev) => new Set(prev).add(file.id));
    setFiles((prev) =>
      prev.map((f) => (f.id === file.id ? { ...f, status: 'uploading' } : f))
    );

    try {
      const result = await singleUploadMutation.mutateAsync({
        file,
        metadata: file.metadata
      } as any);
      console.log('Upload successful:', result);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id
            ? { ...f, status: 'completed', submissionId: result.submissionId }
            : f
        )
      );
    } catch (error: any) {
      console.error('Upload error:', error);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id
            ? {
                ...f,
                status: 'error',
                error: error.response?.data?.message || error.message || 'Upload failed',
              }
            : f
        )
      );
    } finally {
      setUploadingFiles((prev) => {
        const newSet = new Set(prev);
        newSet.delete(file.id);
        return newSet;
      });
    }
  };

  const openMetadataForm = (fileId: string) => {
    const file = files.find(f => f.id === fileId);
    if (file) {
      setCurrentFileId(fileId);
      setMetadata({
        title: file.metadata?.title || file.name.replace(/\.[^/.]+$/, ''),
        courseId: file.metadata?.courseId || '',
        assignmentTitle: file.metadata?.assignmentTitle || '',
        assignmentId: file.metadata?.assignmentId || '',
      });
      setShowMetadataForm(true);
    }
  };

  const saveMetadata = () => {
    if (currentFileId) {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === currentFileId
            ? { ...f, metadata: { ...metadata } }
            : f
        )
      );
      setShowMetadataForm(false);
      setCurrentFileId(null);
    }
  };

  const uploadAllFiles = async () => {
    const pendingFiles = files.filter((f) => f.status === 'pending');
    
    if (isBatchMode) {
      setShowBatchModal(true);
    } else {
      // Upload files individually
      for (const file of pendingFiles) {
        await uploadSingleFile(file);
      }
    }
  };

  const handleBatchUpload = async (metadata: any) => {
    const pendingFiles = files.filter((f) => f.status === 'pending');
    
    try {
      const request: BatchUploadRequest = {
        files: pendingFiles,
        metadata,
      };

      await batchUploadMutation.mutateAsync(request);
      setFiles([]);
      setShowBatchModal(false);
    } catch (error) {
      // Error handled by mutation
    }
  };

  const getFileIcon = (file: File) => {
    const extension = file.name?.split('.').pop()?.toLowerCase();
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'uploading':
      case 'processing':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const pendingFiles = files.filter((f) => f.status === 'pending');
  const hasFiles = files.length > 0;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Essays</h1>
        <p className="text-gray-600 mt-1">
          Upload your essays for authorship verification and blockchain attestation
        </p>
      </div>

      {/* Upload Mode Toggle */}
      <div className="mb-6">
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="radio"
              name="uploadMode"
              checked={!isBatchMode}
              onChange={() => setIsBatchMode(false)}
              className="mr-2"
            />
            <span className="text-sm font-medium text-gray-700">Individual Upload</span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              name="uploadMode"
              checked={isBatchMode}
              onChange={() => setIsBatchMode(true)}
              className="mr-2"
            />
            <span className="text-sm font-medium text-gray-700">Batch Upload</span>
          </label>
        </div>
        <p className="text-sm text-gray-500 mt-1">
          {isBatchMode
            ? 'Upload multiple essays with shared metadata (course, assignment, etc.)'
            : 'Upload essays individually with custom settings for each'}
        </p>
      </div>

      {/* Drop Zone */}
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
          {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
        </p>
        <p className="text-sm text-gray-500">
          or click to select files (PDF, DOCX, TXT up to 10MB each)
        </p>
        {isBatchMode && (
          <div className="mt-2 flex items-center justify-center text-sm text-blue-600">
            <FolderIcon className="h-4 w-4 mr-1" />
            Batch mode: Upload multiple files at once
          </div>
        )}
      </div>

      {/* File List */}
      {hasFiles && (
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium text-gray-900">
              Selected Files ({files.length})
            </h2>
            {pendingFiles.length > 0 && (
              <button
                onClick={uploadAllFiles}
                disabled={singleUploadMutation.isLoading || batchUploadMutation.isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isBatchMode ? 'Upload Batch' : 'Upload All'}
              </button>
            )}
          </div>

          <div className="space-y-3">
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg"
              >
                <div className="flex items-center space-x-3 flex-1">
                  <span className="text-2xl">{getFileIcon(file)}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {file.size ? (file.size / 1024 / 1024).toFixed(2) : '0.00'} MB
                    </p>
                    {file.error && (
                      <p className="text-sm text-red-600 mt-1">{file.error}</p>
                    )}
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  {file.metadata && (file.metadata.courseId || file.metadata.assignmentTitle) && (
                    <div className="text-xs text-gray-500 mr-2">
                      {file.metadata.courseId && <span className="mr-2">📚 {file.metadata.courseId}</span>}
                      {file.metadata.assignmentTitle && <span>📝 {file.metadata.assignmentTitle}</span>}
                    </div>
                  )}

                  {file.status && (
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                        file.status
                      )}`}
                    >
                      {file.status}
                    </span>
                  )}

                  {file.status === 'pending' && !isBatchMode && (
                    <>
                      <button
                        onClick={() => openMetadataForm(file.id)}
                        className="text-gray-600 hover:text-gray-800 text-sm font-medium"
                      >
                        {file.metadata ? '✏️ Edit' : '➕ Add Details'}
                      </button>
                      <button
                        onClick={() => uploadSingleFile(file)}
                        disabled={uploadingFiles.has(file.id)}
                        className="text-blue-600 hover:text-blue-500 text-sm font-medium disabled:opacity-50"
                      >
                        Upload
                      </button>
                    </>
                  )}

                  {file.status !== 'uploading' && file.status !== 'processing' && (
                    <button
                      onClick={() => removeFile(file.id)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <XMarkIcon className="h-5 w-5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Progress Components */}
      {files.some((f) => f.submissionId) && (
        <div className="mt-8">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Processing Status</h2>
          <div className="space-y-4">
            {files
              .filter((f) => f.submissionId)
              .map((file) => (
                <UploadProgress
                  key={file.submissionId}
                  submissionId={file.submissionId!}
                  fileName={file.name}
                />
              ))}
          </div>
        </div>
      )}

      {/* Info Panel */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <InformationCircleIcon className="h-5 w-5 text-blue-400 mt-0.5" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">Upload Guidelines</h3>
            <div className="mt-2 text-sm text-blue-700">
              <ul className="list-disc list-inside space-y-1">
                <li>Supported formats: PDF, DOCX, TXT (max 10MB each)</li>
                <li>Essays will be analyzed for authorship, AI detection, and duplicates</li>
                <li>Verified essays receive blockchain attestation automatically</li>
                <li>Processing typically takes 30-60 seconds per essay</li>
                {isBatchMode && (
                  <li>Batch uploads allow shared metadata for multiple essays</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Metadata Form Modal */}
      {showMetadataForm && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Add Submission Details
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Title
                </label>
                <input
                  type="text"
                  value={metadata.title}
                  onChange={(e) => setMetadata({ ...metadata, title: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Essay title"
                />
                <p className="text-xs text-gray-500 mt-1">Custom title for your submission</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Course ID
                </label>
                <input
                  type="text"
                  value={metadata.courseId}
                  onChange={(e) => setMetadata({ ...metadata, courseId: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., CS101, ENG202"
                />
                <p className="text-xs text-gray-500 mt-1">Course code or identifier</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Assignment Title
                </label>
                <input
                  type="text"
                  value={metadata.assignmentTitle}
                  onChange={(e) => setMetadata({ ...metadata, assignmentTitle: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Essay 1, Final Project"
                />
                <p className="text-xs text-gray-500 mt-1">Name of the assignment</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Assignment ID <span className="text-gray-400">(Optional)</span>
                </label>
                <input
                  type="text"
                  value={metadata.assignmentId}
                  onChange={(e) => setMetadata({ ...metadata, assignmentId: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Assignment reference number"
                />
                <p className="text-xs text-gray-500 mt-1">Internal assignment identifier</p>
              </div>
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowMetadataForm(false);
                  setCurrentFileId(null);
                }}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Cancel
              </button>
              <button
                onClick={saveMetadata}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Save Details
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Batch Upload Modal */}
      {showBatchModal && (
        <BatchUploadModal
          files={pendingFiles}
          onUpload={handleBatchUpload}
          onCancel={() => setShowBatchModal(false)}
          isLoading={batchUploadMutation.isLoading}
        />
      )}
    </div>
  );
};

export default UploadPage;