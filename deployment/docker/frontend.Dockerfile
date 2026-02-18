FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create non-root user
RUN addgroup -g 1001 -S stylos && \
    adduser -S stylos -u 1001

# Set ownership
RUN chown -R stylos:stylos /usr/share/nginx/html && \
    chown -R stylos:stylos /var/cache/nginx && \
    chown -R stylos:stylos /var/log/nginx && \
    chown -R stylos:stylos /etc/nginx/conf.d

# Switch to non-root user
USER stylos

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]