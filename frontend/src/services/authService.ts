import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export interface LoginResponse {
  user: {
    id: string;
    email: string;
    full_name: string;
    role: 'student' | 'admin';
    is_active: boolean;
    is_verified: boolean;
    institution_id?: string;
    student_id?: string;
    created_at: string;
    last_login?: string;
  };
  tokens: {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
  };
}

export interface RegisterResponse {
  token?: string;
  tokens?: {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
  };
  user: {
    id: string;
    email: string;
    name?: string;
    full_name?: string;
    role: 'student' | 'admin';
  };
}

export const authService = {
  async login(email: string, password: string): Promise<LoginResponse> {
    const response = await api.post('/api/v1/auth/login', { email, password });
    return response.data;
  },

  async register(email: string, password: string, name: string): Promise<RegisterResponse> {
    const response = await api.post('/api/v1/auth/register', { email, password, name });
    return response.data;
  },

  async getCurrentUser() {
    const response = await api.get('/api/v1/auth/me');
    return response.data;
  },

  async refreshToken() {
    const response = await api.post('/api/v1/auth/refresh');
    return response.data;
  },
};

export { api };