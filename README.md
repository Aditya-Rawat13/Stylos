# TrueAuthor - Academic Integrity Platform

An AI-powered academic writing verification system with blockchain attestation for ensuring academic authenticity.

## 🚀 Features

- **AI Detection**: Advanced AI-generated content detection using multiple ML models
- **Authorship Verification**: Stylometric analysis for writing profile creation
- **Blockchain Attestation**: Immutable proof-of-authorship on blockchain
- **Duplicate Detection**: Semantic similarity checking across submissions
- **Writing Profile**: Unique fingerprint for each student's writing style
- **Admin Dashboard**: Comprehensive analytics and verification management

## 📋 Prerequisites

### Backend
- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Node.js 16+ (for blockchain)

### Frontend
- Node.js 16+
- npm or yarn

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd prod
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp ../.env.example ../.env
# Edit .env with your configuration

# Initialize database
python scripts/init_database.py

# Run migrations
alembic upgrade head
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local
# Edit .env.local with your API URL

# Build for production
npm run build
```

### 4. Blockchain Setup

```bash
cd blockchain

# Install dependencies
npm install

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Deploy contracts (local)
npx hardhat node  # In separate terminal
npx hardhat run scripts/deploy.js --network localhost
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `prod` directory:

```env
# Database
DATABASE_URL=postgresql://trueauthor_user:your_password@localhost:5432/trueauthor_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Blockchain
POLYGON_RPC_URL=http://127.0.0.1:8545
CONTRACT_ADDRESS=your-contract-address

# Frontend
REACT_APP_API_URL=http://localhost:8000/api/v1
```

## � Running the Application

### Development Mode

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm start

# Terminal 3: Blockchain (optional)
cd blockchain
npx hardhat node

# Terminal 4: Redis
redis-server

# Terminal 5: PostgreSQL
# Make sure PostgreSQL is running
```

### Production Mode with Docker

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📦 Project Structure

```
prod/
├── backend/              # FastAPI backend
│   ├── api/             # API endpoints
│   ├── core/            # Core configuration
│   ├── models/          # Database models
│   ├── services/        # Business logic
│   ├── schemas/         # Pydantic schemas
│   └── tests/           # Test suite
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API services
│   │   └── contexts/    # React contexts
│   └── public/          # Static files
├── blockchain/          # Smart contracts
│   ├── contracts/       # Solidity contracts
│   ├── scripts/         # Deployment scripts
│   └── test/            # Contract tests
├── deployment/          # Kubernetes & monitoring
│   ├── kubernetes/      # K8s manifests
│   ├── monitoring/      # Prometheus/Grafana
│   └── backup/          # Backup scripts
└── database/            # Database scripts
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Blockchain Tests

```bash
cd blockchain
npx hardhat test
```

## 📚 API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔐 Security Notes

1. **Never commit** `.env` files or secrets to Git
2. **Change default passwords** in production
3. **Use HTTPS** in production
4. **Enable CORS** only for trusted domains
5. **Rotate JWT secrets** regularly
6. **Keep dependencies updated**

## 🐳 Docker Deployment

```bash
# Build images
docker-compose build

# Run services
docker-compose up -d

# Scale services
docker-compose up -d --scale backend=3

# View logs
docker-compose logs -f backend
```

## ☸️ Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/backend
```

## 📊 Monitoring

Access monitoring dashboards:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## 🔄 Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Team

- Backend Development
- Frontend Development
- Blockchain Development
- ML/AI Development

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Contact: support@trueauthor.dev

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Mobile application
- [ ] Advanced analytics dashboard
- [ ] Integration with LMS platforms
- [ ] Real-time collaboration features

---

**Note**: This is a production-ready application. Ensure all security measures are in place before deploying to production environments.