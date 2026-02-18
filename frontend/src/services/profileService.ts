import { api } from './authService';

export interface WritingProfile {
  id: string;
  studentId: string;
  stylometricFeatures: StylometricFeatures;
  semanticEmbedding: number[];
  confidenceScore: number;
  sampleCount: number;
  lastUpdated: string;
  createdAt: string;
  statistics: WritingStatistics;
}

export interface StylometricFeatures {
  lexicalRichness: {
    ttr: number; // Type-Token Ratio
    mtld: number; // Measure of Textual Lexical Diversity
    vocdD: number; // Vocabulary Diversity
  };
  syntacticComplexity: {
    avgSentenceLength: number;
    avgClauseLength: number;
    subordinationRatio: number;
  };
  punctuationPatterns: {
    commaFrequency: number;
    semicolonFrequency: number;
    exclamationFrequency: number;
    questionFrequency: number;
  };
  wordFrequencies: {
    functionWords: Record<string, number>;
    contentWords: Record<string, number>;
  };
  posTagDistribution: Record<string, number>;
}

export interface WritingStatistics {
  totalSubmissions: number;
  averageLength: number;
  topicDistribution: Record<string, number>;
  timePatterns: {
    preferredWritingHours: number[];
    averageWritingTime: number;
  };
  improvementMetrics: {
    authorshipConsistency: number;
    styleEvolution: number[];
    qualityTrend: number[];
  };
}

export interface ProfileUpdateRequest {
  samples: File[];
  metadata?: {
    sampleTypes: string[];
    timeframes: string[];
  };
}

export const profileService = {
  async getWritingProfile(): Promise<WritingProfile> {
    const cacheBust = Date.now();
    const response = await api.get(`/api/v1/profile/writing-profile?_cb=${cacheBust}`);
    return response.data;
  },

  async updateWritingProfile(request: ProfileUpdateRequest): Promise<WritingProfile> {
    try {
      const formData = new FormData();
      
      // Validate files before adding to FormData
      request.samples.forEach((file, index) => {
        if (!file || !file.name) {
          throw new Error(`Invalid file at index ${index}`);
        }
        console.log(`Adding file ${index}: ${file.name}, size: ${file.size}, type: ${file.type}`);
        formData.append('samples', file);
      });
      
      if (request.metadata) {
        console.log('Adding metadata:', request.metadata);
        formData.append('metadata', JSON.stringify(request.metadata));
      }

      const cacheBust = Date.now();
      console.log('Sending request to:', `/api/v1/profile/writing/update?_cb=${cacheBust}`);
      
      const response = await api.post(`/api/v1/profile/writing/update?_cb=${cacheBust}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log('Response received:', response.status, response.data);
      return response.data;
    } catch (error: any) {
      console.error('Profile update error:', error);
      if (error.response) {
        console.error('Error response:', error.response.status, error.response.data);
      }
      throw error;
    }
  },

  async getProfileAnalytics(timeRange: '7d' | '30d' | '90d' | '1y' = '30d'): Promise<{
    consistencyTrend: Array<{ date: string; score: number }>;
    featureEvolution: Array<{ date: string; features: Partial<StylometricFeatures> }>;
    comparisonMetrics: {
      peerAverage: number;
      institutionalAverage: number;
      percentile: number;
    };
  }> {
    const cacheBust = Date.now();
    const response = await api.get(`/api/v1/profile/analytics?range=${timeRange}&_cb=${cacheBust}`);
    return response.data;
  },

  async getProfileStrengths(): Promise<{
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
  }> {
    const response = await api.get('/api/v1/profile/strengths');
    return response.data;
  },

  async exportProfile(format: 'json' | 'pdf' | 'csv'): Promise<Blob> {
    const response = await api.get(`/api/v1/profile/export?format=${format}`, {
      responseType: 'blob',
    });
    return response.data;
  },

  async deleteProfile(): Promise<void> {
    await api.delete('/api/v1/profile/writing-profile/reset');
  },

  async getProfileComparison(submissionId: string): Promise<{
    profileMatch: number;
    deviations: Array<{
      feature: string;
      expected: number;
      actual: number;
      significance: 'low' | 'medium' | 'high';
    }>;
    visualComparison: {
      radarChart: Array<{ feature: string; profile: number; submission: number }>;
      timeline: Array<{ date: string; similarity: number }>;
    };
  }> {
    const response = await api.get(`/api/v1/profile/compare/${submissionId}`);
    return response.data;
  },
};