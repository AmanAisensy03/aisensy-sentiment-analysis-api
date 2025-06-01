from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import MessageRequest, BulkMessageRequest, SentimentResponse, BulkSentimentResponse
from sentiment_analyzer import SentimentAnalyzer
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="AiSensy Sentiment Analysis API - Performance Optimized",
    description="Production-ready sentiment analysis for customer chat messages using LangChain + ChatGroq with advanced business intelligence and performance optimization",
    version="1.1.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "message": "AiSensy Sentiment Analysis API - Performance Optimized",
        "version": "1.1.0",
        "status": "healthy",
        "features": [
            "Advanced sentiment analysis",
            "Churn prediction & revenue risk",
            "WhatsApp conversation cost optimization", 
            "Template category recommendations",
            "Performance optimized bulk processing",
            "Multi-language support"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: MessageRequest):
    """
    Analyze sentiment of a single message with comprehensive business intelligence
    
    Args:
        request: MessageRequest containing the message and metadata
    
    Returns:
        SentimentResponse with enhanced sentiment analysis, predictions, and recommendations
    """
    try:
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(request.message) > 2000:
            raise HTTPException(status_code=400, detail="Message too long (max 2000 characters)")
        
        # Add timestamp if not provided
        if not request.timestamp:
            request.timestamp = datetime.now().isoformat()
        
        result = analyzer.analyze_message(
            message=request.message,
            customer_id=request.customer_id,
            agent_id=request.agent_id,
            timestamp=request.timestamp
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/analyze-bulk", response_model=BulkSentimentResponse)
async def analyze_bulk_sentiment(request: BulkMessageRequest):
    """
    Analyze sentiment of multiple messages with performance optimization
    
    Args:
        request: BulkMessageRequest containing list of messages (max 15 for performance)
    
    Returns:
        BulkSentimentResponse with results and enhanced business intelligence summary
    """
    try:
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty")
        
        if len(request.messages) > 15:  # Reduced limit for better performance
            raise HTTPException(
                status_code=400, 
                detail="Maximum 15 messages allowed per bulk request for optimal performance. For larger datasets, use multiple smaller requests."
            )
        
        # Add timestamps for messages that don't have them
        for msg in request.messages:
            if not msg.timestamp:
                msg.timestamp = datetime.now().isoformat()
            if len(msg.message) > 2000:
                msg.message = msg.message[:2000] + "..."
        
        result = analyzer.analyze_bulk_messages(request.messages)
        
        return BulkSentimentResponse(
            results=result["results"],
            summary=result["summary"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk sentiment analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Detailed health check for monitoring
    """
    try:
        # Test the analyzer with a simple message
        test_result = analyzer.analyze_message("Hello, this is a test message for health check")
        
        return {
            "status": "healthy",
            "groq_connection": "working",
            "analyzer_status": "functional",
            "performance_optimizations": "enabled",
            "bulk_processing_limit": 15,
            "single_message_limit": "2000 characters",
            "timestamp": datetime.now().isoformat(),
            "test_sentiment": test_result.sentiment.value,
            "api_version": "1.1.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Enhanced AiSensy integration endpoint
@app.post("/aisensy/chat-analysis")
async def aisensy_chat_analysis(request: MessageRequest):
    """
    Specialized endpoint for AiSensy live chat integration
    Returns simplified response for easier integration with performance optimization
    """
    try:
        # Clean and validate the message
        cleaned_message = request.message.strip().replace('\n', ' ').replace('\r', ' ')
        if len(cleaned_message) > 2000:
            cleaned_message = cleaned_message[:2000] + "..."
            
        if not cleaned_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        result = analyzer.analyze_message(
            message=cleaned_message,
            customer_id=request.customer_id,
            agent_id=request.agent_id,
            timestamp=request.timestamp
        )
        
        # Simplified response for AiSensy integration
        return {
            "message_id": f"msg_{request.customer_id}_{datetime.now().timestamp()}",
            "sentiment": result.sentiment.value,
            "confidence": result.confidence_score,
            "alert_required": result.alert_level in ["medium", "high"],
            "alert_level": result.alert_level,
            "summary": result.reasoning,
            
            # Enhanced business intelligence for AiSensy
            "business_intelligence": {
                "churn_risk": result.churn_probability,
                "revenue_risk": result.revenue_risk,
                "purchase_intent": result.purchase_intent,
                "customer_tier": result.customer_value_tier,
                "retention_action": result.retention_action
            },
            
            # WhatsApp conversation optimization
            "whatsapp_optimization": {
                "recommended_category": result.template_recommendation.primary_category,
                "predicted_cost": result.cost_prediction.predicted_cost,
                "cost_saved": result.cost_prediction.cost_saved,
                "avoid_categories": result.template_recommendation.avoid_categories
            },
            
            # Response guidance
            "response_guidance": {
                "success_probability": result.response_prediction.success_probability,
                "best_response_time": result.response_prediction.best_response_time,
                "escalation_risk": result.response_prediction.escalation_probability
            },
            
            "customer_id": request.customer_id,
            "agent_id": request.agent_id,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AiSensy chat analysis failed: {str(e)}")

# Performance monitoring endpoint
@app.get("/performance")
async def performance_stats():
    """
    API performance statistics and optimization info
    """
    return {
        "api_version": "1.1.0",
        "performance_optimizations": {
            "bulk_processing": "Optimized batch processing for ≤8 messages",
            "individual_fallback": "Progressive processing for larger batches",
            "message_limits": "15 messages max per bulk request",
            "character_limits": "2000 characters max per message",
            "response_time": "~2-5 seconds for single message, ~10-15 seconds for bulk"
        },
        "business_intelligence_features": [
            "Churn probability prediction",
            "Revenue risk assessment", 
            "Purchase intent scoring",
            "WhatsApp conversation cost optimization",
            "Template category recommendations",
            "Response success prediction",
            "Multi-language support (Hindi/English)"
        ],
        "integration_ready": {
            "aisensy_endpoint": "/aisensy/chat-analysis",
            "bulk_endpoint": "/analyze-bulk", 
            "single_endpoint": "/analyze-sentiment",
            "health_check": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)