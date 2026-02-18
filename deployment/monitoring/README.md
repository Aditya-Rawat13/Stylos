# Project Stylos Monitoring and Observability

This directory contains the complete monitoring and observability stack for Project Stylos production deployment.

## Components

### 1. Prometheus (Metrics Collection)
- **Purpose**: Collects and stores time-series metrics
- **Port**: 9090
- **Configuration**: `prometheus.yaml`
- **Features**:
  - Application metrics from backend services
  - Infrastructure metrics from exporters
  - Custom business metrics (submissions, verifications)
  - Alert rules for system health

### 2. Grafana (Visualization)
- **Purpose**: Metrics visualization and dashboards
- **Port**: 3000
- **Configuration**: `grafana.yaml`
- **Features**:
  - Pre-configured dashboards for system overview
  - Real-time monitoring of key metrics
  - Alert visualization
  - User management and access control

### 3. Alertmanager (Alerting)
- **Purpose**: Alert routing and notification
- **Port**: 9093
- **Configuration**: `alertmanager.yaml`
- **Features**:
  - Email notifications for critical alerts
  - Slack integration for team notifications
  - Alert grouping and deduplication
  - Escalation policies

### 4. Exporters (Metrics Sources)
- **PostgreSQL Exporter**: Database metrics
- **Redis Exporter**: Cache metrics
- **Node Exporter**: System metrics
- **Configuration**: `exporters.yaml`

### 5. ELK Stack (Logging)
- **Elasticsearch**: Log storage and indexing
- **Kibana**: Log visualization and analysis
- **Fluentd**: Log collection and forwarding
- **Configuration**: `logging.yaml`

## Deployment

### Prerequisites
- Kubernetes cluster with sufficient resources
- kubectl configured and connected
- Required environment variables set

### Environment Variables
```bash
# Required
export DB_PASSWORD="your_database_password"
export GRAFANA_ADMIN_PASSWORD="secure_grafana_password"

# Optional
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### Deploy Monitoring Stack
```powershell
# Deploy complete monitoring stack
.\prod\deployment\scripts\deploy-monitoring.ps1

# Deploy without alerting
.\prod\deployment\scripts\deploy-monitoring.ps1 -SkipAlerts

# Deploy to specific namespace
.\prod\deployment\scripts\deploy-monitoring.ps1 -Namespace "stylos-staging"
```

### Manual Deployment
```bash
# Deploy components individually
kubectl apply -f prod/deployment/monitoring/prometheus.yaml -n stylos-prod
kubectl apply -f prod/deployment/monitoring/grafana.yaml -n stylos-prod
kubectl apply -f prod/deployment/monitoring/exporters.yaml -n stylos-prod
kubectl apply -f prod/deployment/monitoring/alertmanager.yaml -n stylos-prod
kubectl apply -f prod/deployment/monitoring/logging.yaml -n stylos-prod
```

## Access

### Port Forwarding
```bash
# Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n stylos-prod

# Grafana
kubectl port-forward svc/grafana 3000:3000 -n stylos-prod

# Kibana
kubectl port-forward svc/kibana 5601:5601 -n stylos-prod

# Alertmanager
kubectl port-forward svc/alertmanager 9093:9093 -n stylos-prod
```

### URLs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
- Kibana: http://localhost:5601
- Alertmanager: http://localhost:9093

## Key Metrics

### Application Metrics
- `http_requests_total`: Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds`: Request latency histogram
- `http_requests_active`: Current active requests
- `active_users_total`: Number of active users
- `submissions_total`: Total submissions processed
- `verification_duration_seconds`: Verification processing time
- `blockchain_operations_total`: Blockchain operations count

### Infrastructure Metrics
- `up`: Service availability
- `container_memory_usage_bytes`: Memory usage
- `container_cpu_usage_seconds_total`: CPU usage
- `pg_stat_database_numbackends`: Database connections
- `redis_connected_clients`: Redis connections

### Business Metrics
- Submission success/failure rates
- Verification accuracy metrics
- User engagement metrics
- System performance trends

## Alerts

### Critical Alerts
- **DatabaseConnectionFailure**: Database unavailable
- **HighErrorRate**: Error rate > 10%
- **PodCrashLooping**: Pods restarting frequently

### Warning Alerts
- **HighLatency**: 95th percentile > 500ms
- **HighMemoryUsage**: Memory usage > 90%
- **HighCPUUsage**: CPU usage > 80%
- **RedisConnectionFailure**: Redis unavailable

## Dashboards

### System Overview Dashboard
- Request rate and latency
- Error rates and status codes
- Active users and sessions
- Resource utilization

### Infrastructure Dashboard
- Pod status and health
- Resource usage trends
- Database performance
- Cache hit rates

### Business Dashboard
- Submission metrics
- Verification performance
- User activity patterns
- Revenue/usage trends

## Log Analysis

### Log Sources
- Application logs (structured JSON)
- Access logs (nginx/ingress)
- System logs (kubernetes events)
- Audit logs (security events)

### Log Queries
```
# Find errors in last hour
level:ERROR AND @timestamp:[now-1h TO now]

# Search for specific user activity
user_id:"12345" AND action:"submission"

# Find slow requests
response_time:>1000 AND @timestamp:[now-15m TO now]
```

## Troubleshooting

### Common Issues

#### Prometheus Not Scraping Targets
```bash
# Check service discovery
kubectl get endpoints -n stylos-prod

# Check pod annotations
kubectl describe pod <pod-name> -n stylos-prod

# Verify metrics endpoint
kubectl exec -it <pod-name> -n stylos-prod -- curl localhost:8000/metrics
```

#### Grafana Dashboard Not Loading
```bash
# Check Grafana logs
kubectl logs deployment/grafana -n stylos-prod

# Verify datasource connection
kubectl exec -it deployment/grafana -n stylos-prod -- curl prometheus:9090/api/v1/query?query=up
```

#### Elasticsearch Issues
```bash
# Check cluster health
kubectl exec -it elasticsearch-0 -n stylos-prod -- curl localhost:9200/_cluster/health

# Check indices
kubectl exec -it elasticsearch-0 -n stylos-prod -- curl localhost:9200/_cat/indices
```

### Performance Tuning

#### Prometheus
- Adjust retention period: `--storage.tsdb.retention.time=30d`
- Increase memory: Update resource limits
- Optimize queries: Use recording rules for complex queries

#### Grafana
- Enable caching: Set `GF_RENDERING_SERVER_URL`
- Optimize dashboards: Reduce query frequency
- Use variables: Parameterize dashboards

#### Elasticsearch
- Adjust heap size: Set `ES_JAVA_OPTS="-Xms2g -Xmx2g"`
- Configure shards: Optimize index settings
- Set up ILM: Implement index lifecycle management

## Maintenance

### Regular Tasks
- Monitor disk usage for metrics storage
- Review and update alert thresholds
- Clean up old logs and metrics
- Update dashboard configurations
- Test alert notifications

### Backup
```bash
# Backup Prometheus data
kubectl exec prometheus-0 -n stylos-prod -- tar czf /tmp/prometheus-backup.tar.gz /prometheus

# Backup Grafana dashboards
kubectl get configmap grafana-dashboard-stylos -n stylos-prod -o yaml > grafana-backup.yaml
```

### Updates
```bash
# Update Prometheus
kubectl set image deployment/prometheus prometheus=prom/prometheus:v2.46.0 -n stylos-prod

# Update Grafana
kubectl set image deployment/grafana grafana=grafana/grafana:10.1.0 -n stylos-prod
```

## Security

### Access Control
- Use RBAC for service accounts
- Implement network policies
- Enable authentication for Grafana
- Secure Elasticsearch with authentication

### Data Protection
- Encrypt metrics at rest
- Use TLS for all communications
- Implement log anonymization
- Regular security audits

## Support

For issues with the monitoring stack:
1. Check component logs: `kubectl logs <pod-name> -n stylos-prod`
2. Verify resource usage: `kubectl top pods -n stylos-prod`
3. Check service connectivity: `kubectl get endpoints -n stylos-prod`
4. Review configuration: `kubectl describe configmap <config-name> -n stylos-prod`