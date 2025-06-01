from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class MessageRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: Optional[str] = None

class BulkMessageRequest(BaseModel):
    messages: List[MessageRequest]

class CostPrediction(BaseModel):
    optimal_conversation_type: str
    predicted_cost: str
    cost_saved: str
    reasoning: str

class ResponsePrediction(BaseModel):
    success_probability: int
    best_response_time: str
    escalation_probability: int
    resolution_likelihood: str

class TemplateRecommendation(BaseModel):
    primary_category: str
    confidence: int
    cost_impact: str
    avoid_categories: List[str]
    reasoning: str

class SentimentResponse(BaseModel):
    message: str
    sentiment: SentimentType
    confidence_score: float
    reasoning: str
    customer_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: Optional[str] = None
    alert_level: str
    
    # Business Intelligence Parameters
    churn_probability: int
    revenue_risk: str
    purchase_intent: int
    customer_value_tier: str
    retention_action: str
    
    # Prediction Engine
    cost_prediction: CostPrediction
    response_prediction: ResponsePrediction
    template_recommendation: TemplateRecommendation

class BulkSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    summary: dict