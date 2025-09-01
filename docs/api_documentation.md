# API Documentation - Retail Customer Analytics

## Overview

This document provides comprehensive API documentation for the Retail Customer Analytics platform. The system offers programmatic access to customer analytics, segmentation, predictions, and recommendations.

## Table of Contents

1. [Authentication](#authentication)
2. [Base URLs](#base-urls)
3. [Data Ingestion API](#data-ingestion-api)
4. [Analytics API](#analytics-api)
5. [Prediction API](#prediction-api)
6. [Recommendation API](#recommendation-api)
7. [Model Management API](#model-management-api)
8. [Error Handling](#error-handling)
9. [SDKs and Examples](#sdks-and-examples)

## Authentication

The API uses API key authentication. Include your API key in the request headers:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

## Base URLs

- **Production**: `https://api.retailanalytics.com/v1`
- **Staging**: `https://staging-api.retailanalytics.com/v1`
- **Development**: `http://localhost:8000/v1`

## Data Ingestion API

### Upload Customer Data

Upload customer transaction data for analysis.

**Endpoint:** `POST /data/upload`

**Parameters:**
- `file` (file): CSV file containing customer data
- `data_type` (string): Type of data ('transactions', 'customers', 'products')
- `format` (string): Data format ('csv', 'json', 'parquet')

**Example Request:**
```bash
curl -X POST "https://api.retailanalytics.com/v1/data/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@customer_data.csv" \
  -F "data_type=transactions" \
  -F "format=csv"
```

**Response:**
```json
{
  "status": "success",
  "upload_id": "upload_123456",
  "records_processed": 15000,
  "data_quality_score": 0.95,
  "message": "Data uploaded and processed successfully"
}
```

### Get Data Schema

Retrieve the expected data schema for uploads.

**Endpoint:** `GET /data/schema`

**Parameters:**
- `data_type` (string): Type of schema to retrieve

**Example Request:**
```bash
curl -X GET "https://api.retailanalytics.com/v1/data/schema?data_type=transactions" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "schema": {
    "required_fields": [
      {"name": "customer_id", "type": "string", "description": "Unique customer identifier"},
      {"name": "purchase_date", "type": "datetime", "description": "Date of purchase"},
      {"name": "purchase_amount", "type": "float", "description": "Purchase amount in USD"}
    ],
    "optional_fields": [
      {"name": "category", "type": "string", "description": "Product category"},
      {"name": "review_rating", "type": "float", "description": "Customer rating 1-5"}
    ]
  }
}
```

## Analytics API

### Customer Segmentation

Perform customer segmentation analysis.

**Endpoint:** `POST /analytics/segmentation`

**Parameters:**
```json
{
  "dataset_id": "dataset_123",
  "algorithm": "kmeans",
  "n_clusters": 5,
  "features": ["recency", "frequency", "monetary"],
  "config": {
    "random_state": 42,
    "max_iter": 300
  }
}
```

**Example Request:**
```bash
curl -X POST "https://api.retailanalytics.com/v1/analytics/segmentation" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset_123",
    "algorithm": "kmeans",
    "n_clusters": 5
  }'
```

**Response:**
```json
{
  "segmentation_id": "seg_789",
  "status": "completed",
  "segments": {
    "0": {
      "name": "High Value Customers",
      "size": 1250,
      "characteristics": {
        "avg_purchase_amount": 245.50,
        "avg_frequency": 8.2,
        "avg_recency": 15.3
      }
    },
    "1": {
      "name": "At-Risk Customers", 
      "size": 890,
      "characteristics": {
        "avg_purchase_amount": 89.20,
        "avg_frequency": 2.1,
        "avg_recency": 120.5
      }
    }
  },
  "performance_metrics": {
    "silhouette_score": 0.72,
    "inertia": 1250.8
  }
}
```

### RFM Analysis

Perform RFM (Recency, Frequency, Monetary) analysis.

**Endpoint:** `POST /analytics/rfm`

**Parameters:**
```json
{
  "dataset_id": "dataset_123",
  "customer_col": "customer_id",
  "date_col": "purchase_date", 
  "amount_col": "purchase_amount",
  "quantiles": 5
}
```

**Response:**
```json
{
  "analysis_id": "rfm_456",
  "status": "completed",
  "summary": {
    "total_customers": 5000,
    "avg_recency": 45.2,
    "avg_frequency": 3.8,
    "avg_monetary": 156.30
  },
  "segments": {
    "Champions": {
      "criteria": "R:5, F:5, M:5",
      "count": 250,
      "percentage": 5.0
    },
    "Loyal Customers": {
      "criteria": "R:4-5, F:3-5, M:3-5",
      "count": 800,
      "percentage": 16.0
    }
  }
}
```

## Prediction API

### Churn Prediction

Predict customer churn probability.

**Endpoint:** `POST /predictions/churn`

**Parameters:**
```json
{
  "customer_ids": ["cust_001", "cust_002"],
  "model_version": "v1.2.3",
  "features": {
    "recency": [30, 60],
    "frequency": [5, 2],
    "monetary": [500, 150]
  }
}
```

**Example Request:**
```bash
curl -X POST "https://api.retailanalytics.com/v1/predictions/churn" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_ids": ["cust_001", "cust_002"],
    "model_version": "v1.2.3"
  }'
```

**Response:**
```json
{
  "prediction_id": "pred_789",
  "status": "completed",
  "predictions": [
    {
      "customer_id": "cust_001",
      "churn_probability": 0.15,
      "risk_level": "low",
      "confidence": 0.89
    },
    {
      "customer_id": "cust_002", 
      "churn_probability": 0.78,
      "risk_level": "high",
      "confidence": 0.92
    }
  ],
  "model_info": {
    "version": "v1.2.3",
    "accuracy": 0.87,
    "last_trained": "2024-01-15T10:30:00Z"
  }
}
```

### Customer Lifetime Value (CLV)

Predict customer lifetime value.

**Endpoint:** `POST /predictions/clv`

**Parameters:**
```json
{
  "customer_ids": ["cust_001", "cust_002"],
  "time_horizon": "12_months",
  "model_version": "latest"
}
```

**Response:**
```json
{
  "prediction_id": "clv_123",
  "predictions": [
    {
      "customer_id": "cust_001",
      "predicted_clv": 1250.50,
      "confidence_interval": [1100.00, 1400.00],
      "quartile": "high_value"
    },
    {
      "customer_id": "cust_002",
      "predicted_clv": 340.75,
      "confidence_interval": [280.00, 420.00], 
      "quartile": "medium_value"
    }
  ]
}
```

## Recommendation API

### Product Recommendations

Get personalized product recommendations for customers.

**Endpoint:** `POST /recommendations/products`

**Parameters:**
```json
{
  "customer_id": "cust_001",
  "n_recommendations": 5,
  "method": "collaborative_filtering",
  "filters": {
    "category": ["electronics", "books"],
    "price_range": [10, 200]
  }
}
```

**Example Request:**
```bash
curl -X POST "https://api.retailanalytics.com/v1/recommendations/products" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_001",
    "n_recommendations": 5,
    "method": "collaborative_filtering"
  }'
```

**Response:**
```json
{
  "recommendation_id": "rec_456",
  "customer_id": "cust_001",
  "recommendations": [
    {
      "product_id": "prod_123",
      "product_name": "Wireless Headphones",
      "category": "electronics",
      "predicted_rating": 4.2,
      "confidence": 0.85,
      "reason": "Customers like you also purchased"
    },
    {
      "product_id": "prod_456", 
      "product_name": "Programming Book",
      "category": "books",
      "predicted_rating": 4.0,
      "confidence": 0.78,
      "reason": "Based on your purchase history"
    }
  ],
  "method_used": "collaborative_filtering",
  "generated_at": "2024-01-15T14:30:00Z"
}
```

### Similar Customers

Find customers similar to a given customer.

**Endpoint:** `GET /recommendations/similar-customers/{customer_id}`

**Parameters:**
- `n_similar` (int): Number of similar customers to return (default: 10)
- `similarity_metric` (string): Metric to use ('cosine', 'euclidean', 'jaccard')

**Response:**
```json
{
  "customer_id": "cust_001",
  "similar_customers": [
    {
      "customer_id": "cust_045",
      "similarity_score": 0.94,
      "common_categories": ["electronics", "books", "clothing"]
    },
    {
      "customer_id": "cust_078",
      "similarity_score": 0.87,
      "common_categories": ["electronics", "sports"]
    }
  ]
}
```

## Model Management API

### List Models

Get list of available models.

**Endpoint:** `GET /models`

**Response:**
```json
{
  "models": [
    {
      "model_id": "churn_v1.2.3",
      "model_type": "churn_prediction",
      "version": "v1.2.3",
      "status": "active",
      "accuracy": 0.87,
      "last_trained": "2024-01-15T10:30:00Z",
      "training_data_size": 50000
    },
    {
      "model_id": "clv_v2.1.0",
      "model_type": "clv_prediction", 
      "version": "v2.1.0",
      "status": "active",
      "r2_score": 0.82,
      "last_trained": "2024-01-10T15:45:00Z",
      "training_data_size": 35000
    }
  ]
}
```

### Retrain Model

Trigger model retraining with new data.

**Endpoint:** `POST /models/{model_id}/retrain`

**Parameters:**
```json
{
  "dataset_id": "dataset_123",
  "hyperparameters": {
    "learning_rate": 0.01,
    "n_estimators": 100
  },
  "validation_split": 0.2
}
```

**Response:**
```json
{
  "job_id": "train_job_789",
  "status": "started",
  "estimated_completion": "2024-01-15T16:00:00Z",
  "message": "Model retraining initiated"
}
```

## Error Handling

The API uses conventional HTTP status codes to indicate success or failure:

- `200` - OK: Request successful
- `201` - Created: Resource created successfully  
- `400` - Bad Request: Invalid request parameters
- `401` - Unauthorized: Invalid API key
- `403` - Forbidden: Insufficient permissions
- `404` - Not Found: Resource not found
- `429` - Too Many Requests: Rate limit exceeded
- `500` - Internal Server Error: Server error

**Error Response Format:**
```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "The parameter 'n_clusters' must be between 2 and 20",
    "details": {
      "parameter": "n_clusters",
      "provided_value": 1,
      "valid_range": [2, 20]
    },
    "request_id": "req_123456"
  }
}
```

## Rate Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: Custom limits

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## SDKs and Examples

### Python SDK

```python
from retail_analytics import RetailAnalyticsClient

# Initialize client
client = RetailAnalyticsClient(api_key="your_api_key")

# Upload data
upload_result = client.data.upload("customer_data.csv", data_type="transactions")

# Perform segmentation
segmentation = client.analytics.segment_customers(
    dataset_id=upload_result.dataset_id,
    algorithm="kmeans",
    n_clusters=5
)

# Get churn predictions
predictions = client.predictions.predict_churn(
    customer_ids=["cust_001", "cust_002"]
)

# Get recommendations
recommendations = client.recommendations.get_products(
    customer_id="cust_001",
    n_recommendations=5
)
```

### JavaScript SDK

```javascript
import RetailAnalytics from 'retail-analytics-js';

const client = new RetailAnalytics({
  apiKey: 'your_api_key',
  baseURL: 'https://api.retailanalytics.com/v1'
});

// Upload data
const uploadResult = await client.data.upload({
  file: fileData,
  dataType: 'transactions',
  format: 'csv'
});

// Perform segmentation
const segmentation = await client.analytics.segmentCustomers({
  datasetId: uploadResult.datasetId,
  algorithm: 'kmeans',
  nClusters: 5
});

// Get predictions
const predictions = await client.predictions.predictChurn({
  customerIds: ['cust_001', 'cust_002']
});
```

### cURL Examples

**Upload Data:**
```bash
curl -X POST "https://api.retailanalytics.com/v1/data/upload" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@data.csv" \
  -F "data_type=transactions"
```

**Get Recommendations:**
```bash
curl -X POST "https://api.retailanalytics.com/v1/recommendations/products" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cust_001", "n_recommendations": 5}'
```

## Webhook Support

Configure webhooks to receive notifications for long-running operations:

**Endpoint:** `POST /webhooks`

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["training_completed", "prediction_ready"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "training_completed",
  "timestamp": "2024-01-15T16:00:00Z",
  "data": {
    "job_id": "train_job_789",
    "model_id": "churn_v1.2.4", 
    "status": "completed",
    "metrics": {
      "accuracy": 0.89,
      "precision": 0.87,
      "recall": 0.85
    }
  }
}
```

## Support

For API support and questions:

- **Documentation**: https://docs.retailanalytics.com
- **Support Email**: api-support@retailanalytics.com  
- **Status Page**: https://status.retailanalytics.com
- **Community Forum**: https://forum.retailanalytics.com

---

**Last Updated**: January 15, 2024  
**API Version**: v1.0  
**SDK Versions**: Python v1.2.0, JavaScript v1.1.0