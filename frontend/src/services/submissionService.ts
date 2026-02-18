import { api } from './authService';

export interface Submission {
  id: string;
  title: string;
  content: string;
  contentHash: string;
  ipfsHash?: string;
  verificationResult: VerificationResult;
  blockchainTxHash?: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'REVIEW' | 'pending' | 'processing' | 'completed' | 'failed' | 'review' | 'uploaded' | 'UPLOADED' | 'verified' | 'VERIFIED' | 'approved' | 'APPROVED' | 'flagged' | 'FLAGGED' | 'rejected' | 'REJECTED';
  submittedAt: string;
  updatedAt: string;
}

export interface VerificationResult {
  authorshipScore: number;
  aiProbability: number;
  duplicateMatches: DuplicateMatch[];
  overallStatus: 'PASS' | 'FAIL' | 'REVIEW';
  confidence: number;
  processingTime?: number;
  details?: {
    stylometricFeatures?: any;
    semanticSimilarity?: number;
    flaggedReasons?: string[];
  };
}

export interface DuplicateMatch {
  submissionId: string;
  similarityScore: number;
  matchType: 'EXACT' | 'NEAR_DUPLICATE' | 'SEMANTIC';
  studentId: string;
  studentName?: string;
  courseName?: string;
  submittedAt: string;
}

export interface UploadProgress {
  submissionId: string;
  stage: 'UPLOADING' | 'PROCESSING' | 'ANALYZING' | 'BLOCKCHAIN' | 'COMPLETED';
  progress: number;
  message: string;
  estimatedTimeRemaining?: number;
}

export interface BatchUploadRequest {
  files: File[];
  metadata: {
    course?: string;
    assignment?: string;
    dueDate?: string;
  };
}

export const submissionService = {
  async uploadSubmission(file: File, metadata?: {
    title?: string;
    courseId?: string;
    assignmentTitle?: string;
    assignmentId?: string;
  }): Promise<{ submissionId: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add metadata as separate form fields
    if (metadata) {
      if (metadata.title) formData.append('title', metadata.title);
      if (metadata.courseId) formData.append('course_id', metadata.courseId);
      if (metadata.assignmentTitle) formData.append('assignment_title', metadata.assignmentTitle);
      if (metadata.assignmentId) formData.append('assignment_id', metadata.assignmentId);
    }

    const response = await api.post('/api/v1/submissions/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async batchUpload(request: BatchUploadRequest): Promise<{ submissionIds: string[] }> {
    const formData = new FormData();
    request.files.forEach((file, index) => {
      formData.append(`files`, file);
    });
    formData.append('metadata', JSON.stringify(request.metadata));

    const response = await api.post('/api/v1/submissions/batch-upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getSubmissions(page = 1, limit = 10, status?: string): Promise<{
    submissions: Submission[];
    total: number;
    page: number;
    totalPages: number;
  }> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });
    if (status) {
      params.append('status', status);
    }

    const response = await api.get(`/api/v1/submissions?${params}`);
    return response.data;
  },

  async getSubmission(id: string): Promise<Submission> {
    const response = await api.get(`/api/v1/submissions/${id}`);
    return response.data;
  },

  async getUploadProgress(submissionId: string): Promise<UploadProgress> {
    const response = await api.get(`/api/v1/submissions/${submissionId}/progress`);
    return response.data;
  },

  async retryVerification(submissionId: string): Promise<void> {
    await api.post(`/api/v1/submissions/${submissionId}/retry`);
  },

  async deleteSubmission(submissionId: string): Promise<void> {
    await api.delete(`/api/v1/submissions/${submissionId}`);
  },

  async getVerificationReport(submissionId: string): Promise<{
    submission: Submission;
    detailedAnalysis: any;
    recommendations: string[];
  }> {
    const response = await api.get(`/api/v1/submissions/${submissionId}/report`);
    return response.data;
  },

  // Real-time status updates using Server-Sent Events
  subscribeToUpdates(submissionId: string, onUpdate: (progress: UploadProgress) => void): EventSource {
    // Get token from localStorage
    const token = localStorage.getItem('token');
    
    // EventSource doesn't support Authorization headers, so pass token as query parameter
    const url = token 
      ? `${api.defaults.baseURL}/api/v1/submissions/${submissionId}/stream?token=${encodeURIComponent(token)}`
      : `${api.defaults.baseURL}/api/v1/submissions/${submissionId}/stream`;
    
    const eventSource = new EventSource(url, {
      withCredentials: true,
    });

    eventSource.onmessage = (event) => {
      const progress = JSON.parse(event.data);
      onUpdate(progress);
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      eventSource.close();
    };

    return eventSource;
  },
};