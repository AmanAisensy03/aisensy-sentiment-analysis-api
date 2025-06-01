import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from models import SentimentResponse, SentimentType, CostPrediction, ResponsePrediction, TemplateRecommendation
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
            temperature=0.1,
            max_tokens=2000  # Increased for bulk processing
        )
        
        # Enhanced single message prompt
        self.prompt_template = PromptTemplate(
            input_variables=["message"],
            template="""
            You are an expert sentiment analyzer for WhatsApp Business conversations with advanced prediction and recommendation capabilities.
            Analyze the message and provide comprehensive business intelligence including cost optimization and template recommendations.
            
            Message: "{message}"
            
            Respond in this EXACT JSON format:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence_score": 0.0-1.0,
                "reasoning": "Brief explanation of sentiment classification",
                "alert_level": "low/medium/high",
                
                "churn_probability": 0-100,
                "revenue_risk": "safe/at_risk/high_risk/critical",
                "purchase_intent": 0-100,
                "customer_value_tier": "low_value/medium_value/high_value/vip",
                "retention_action": "none/follow_up/discount_offer/manager_call/urgent_intervention",
                
                "cost_prediction": {{
                    "optimal_conversation_type": "service/utility/marketing/authentication",
                    "predicted_cost": "₹0.00/₹0.125/₹0.88",
                    "cost_saved": "₹0.00/₹0.125/₹0.88",
                    "reasoning": "Why this conversation type is recommended"
                }},
                
                "response_prediction": {{
                    "success_probability": 0-100,
                    "best_response_time": "immediate/within_1_hour/within_24_hours",
                    "escalation_probability": 0-100,
                    "resolution_likelihood": "low/medium/high"
                }},
                
                "template_recommendation": {{
                    "primary_category": "service/utility/marketing/authentication",
                    "confidence": 0-100,
                    "cost_impact": "₹0.00/₹0.125/₹0.88",
                    "avoid_categories": ["list", "of", "categories"],
                    "reasoning": "Why this template category is recommended"
                }}
            }}
            
            BUSINESS RULES:
            1. Negative sentiment → service category (FREE) - focus on problem resolution
            2. Positive sentiment + high purchase intent → marketing category (₹0.88) - conversion opportunity  
            3. Neutral sentiment → utility category (₹0.125) - informational updates
            4. Never recommend marketing to angry/frustrated customers - waste of money
            5. High churn risk customers → immediate response required
            6. High-value customers → prioritize retention actions
            7. Authentication only for login/security related messages
            
            Only respond with valid JSON, no additional text.
            """
        )
        
        # Optimized bulk processing prompt
        self.bulk_prompt_template = PromptTemplate(
            input_variables=["messages_batch"],
            template="""
            You are an expert sentiment analyzer. Analyze these messages and return a JSON array with analysis for each message.
            
            Messages to analyze:
            {messages_batch}
            
            For each message, provide this exact structure in a JSON array:
            [
                {{
                    "message_index": 1,
                    "sentiment": "positive/negative/neutral",
                    "confidence_score": 0.0-1.0,
                    "reasoning": "Brief explanation",
                    "alert_level": "low/medium/high",
                    "churn_probability": 0-100,
                    "revenue_risk": "safe/at_risk/high_risk/critical",
                    "purchase_intent": 0-100,
                    "customer_value_tier": "low_value/medium_value/high_value/vip",
                    "retention_action": "none/follow_up/discount_offer/manager_call/urgent_intervention",
                    "cost_prediction": {{
                        "optimal_conversation_type": "service/utility/marketing",
                        "predicted_cost": "₹0.00/₹0.125/₹0.88",
                        "cost_saved": "₹0.00/₹0.125/₹0.88",
                        "reasoning": "Brief reason"
                    }},
                    "response_prediction": {{
                        "success_probability": 0-100,
                        "best_response_time": "immediate/within_1_hour/within_24_hours",
                        "escalation_probability": 0-100,
                        "resolution_likelihood": "low/medium/high"
                    }},
                    "template_recommendation": {{
                        "primary_category": "service/utility/marketing",
                        "confidence": 0-100,
                        "cost_impact": "₹0.00/₹0.125/₹0.88",
                        "avoid_categories": ["list"],
                        "reasoning": "Brief reason"
                    }}
                }}
            ]
            
            Rules: Negative→service(₹0.00), Positive→marketing(₹0.88), Neutral→utility(₹0.125)
            Only return valid JSON array, no additional text.
            """
        )
    
    def analyze_message(self, message: str, customer_id: str = None, agent_id: str = None, timestamp: str = None) -> SentimentResponse:
        """
        Enhanced sentiment analysis with prediction and recommendation engine
        """
        try:
            # Create the prompt
            formatted_prompt = self.prompt_template.format(message=message)
            
            # Get response from Groq
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse the JSON response
            response_text = response.content.strip()
            
            # Clean the response if it has extra text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            parsed_response = json.loads(response_text)
            
            # Create enhanced SentimentResponse object
            sentiment_response = SentimentResponse(
                message=message,
                sentiment=SentimentType(parsed_response["sentiment"].lower()),
                confidence_score=float(parsed_response["confidence_score"]),
                reasoning=parsed_response["reasoning"],
                alert_level=parsed_response["alert_level"].lower(),
                
                # Business Intelligence Parameters
                churn_probability=int(parsed_response["churn_probability"]),
                revenue_risk=parsed_response["revenue_risk"],
                purchase_intent=int(parsed_response["purchase_intent"]),
                customer_value_tier=parsed_response["customer_value_tier"],
                retention_action=parsed_response["retention_action"],
                
                # Prediction Engine
                cost_prediction=CostPrediction(**parsed_response["cost_prediction"]),
                response_prediction=ResponsePrediction(**parsed_response["response_prediction"]),
                template_recommendation=TemplateRecommendation(**parsed_response["template_recommendation"]),
                
                customer_id=customer_id,
                agent_id=agent_id,
                timestamp=timestamp
            )
            
            return sentiment_response
            
        except Exception as e:
            # Fallback response if analysis fails
            return self._create_fallback_response(message, customer_id, agent_id, timestamp, f"Analysis error: {str(e)}")
    
    def analyze_bulk_messages(self, messages: list) -> dict:
        """
        OPTIMIZED bulk analysis - faster processing with intelligent batching
        """
        # Limit messages for performance (max 15 for demo)
        if len(messages) > 15:
            messages = messages[:15]
            print(f"⚠️ Limited to first 15 messages for performance")
        
        # For small batches (≤ 8), use optimized batch processing
        if len(messages) <= 8:
            return self._analyze_bulk_optimized(messages)
        
        # For larger batches, use individual processing with progress
        else:
            return self._analyze_bulk_individual(messages)
    
    def _analyze_bulk_optimized(self, messages: list) -> dict:
        """
        Optimized batch processing for small message sets (≤ 8 messages)
        Single API call for better performance
        """
        try:
            # Prepare batch of messages
            messages_batch = ""
            for i, msg_request in enumerate(messages, 1):
                messages_batch += f"Message {i}: {msg_request.message}\n"
            
            # Single API call for all messages
            formatted_prompt = self.bulk_prompt_template.format(messages_batch=messages_batch)
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse bulk response
            response_text = response.content.strip()
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            parsed_responses = json.loads(response_text)
            
            # Create SentimentResponse objects
            results = []
            for i, msg_request in enumerate(messages):
                if i < len(parsed_responses):
                    parsed_data = parsed_responses[i]
                    sentiment_response = SentimentResponse(
                        message=msg_request.message,
                        sentiment=SentimentType(parsed_data["sentiment"].lower()),
                        confidence_score=float(parsed_data["confidence_score"]),
                        reasoning=parsed_data["reasoning"],
                        alert_level=parsed_data["alert_level"].lower(),
                        
                        churn_probability=int(parsed_data["churn_probability"]),
                        revenue_risk=parsed_data["revenue_risk"],
                        purchase_intent=int(parsed_data["purchase_intent"]),
                        customer_value_tier=parsed_data["customer_value_tier"],
                        retention_action=parsed_data["retention_action"],
                        
                        cost_prediction=CostPrediction(**parsed_data["cost_prediction"]),
                        response_prediction=ResponsePrediction(**parsed_data["response_prediction"]),
                        template_recommendation=TemplateRecommendation(**parsed_data["template_recommendation"]),
                        
                        customer_id=msg_request.customer_id,
                        agent_id=msg_request.agent_id,
                        timestamp=msg_request.timestamp
                    )
                    results.append(sentiment_response)
                else:
                    # Fallback for missing responses
                    results.append(self._create_fallback_response(
                        msg_request.message, msg_request.customer_id, 
                        msg_request.agent_id, msg_request.timestamp, "Batch processing incomplete"
                    ))
            
            return self._calculate_summary(results)
            
        except Exception as e:
            print(f"Batch processing failed: {e}, falling back to individual processing")
            return self._analyze_bulk_individual(messages)
    
    def _analyze_bulk_individual(self, messages: list) -> dict:
        """
        Individual message processing with progress tracking
        """
        results = []
        print(f"🔄 Processing {len(messages)} messages individually...")
        
        for i, msg_request in enumerate(messages, 1):
            try:
                print(f"Processing message {i}/{len(messages)}...")
                result = self.analyze_message(
                    message=msg_request.message,
                    customer_id=msg_request.customer_id,
                    agent_id=msg_request.agent_id,
                    timestamp=msg_request.timestamp
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing message {i}: {e}")
                results.append(self._create_fallback_response(
                    msg_request.message, msg_request.customer_id,
                    msg_request.agent_id, msg_request.timestamp, f"Processing error: {str(e)}"
                ))
        
        return self._calculate_summary(results)
    
    def _calculate_summary(self, results: list) -> dict:
        """
        Calculate enhanced summary statistics
        """
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        alert_counts = {"low": 0, "medium": 0, "high": 0}
        
        # Business intelligence aggregation
        total_churn_risk = 0
        total_purchase_intent = 0
        high_value_customers = 0
        cost_optimization_savings = 0
        
        for result in results:
            # Count sentiments and alerts
            sentiment_counts[result.sentiment.value] += 1
            alert_counts[result.alert_level] += 1
            
            # Aggregate business intelligence
            total_churn_risk += result.churn_probability
            total_purchase_intent += result.purchase_intent
            if result.customer_value_tier in ["high_value", "vip"]:
                high_value_customers += 1
            
            # Calculate cost savings
            cost_saved_text = result.cost_prediction.cost_saved.replace("₹", "")
            try:
                cost_saved_value = float(cost_saved_text)
                cost_optimization_savings += cost_saved_value
            except:
                pass
        
        # Calculate enhanced summary statistics
        total_messages = len(results)
        summary = {
            "total_messages": total_messages,
            "processing_time": "Optimized for performance",
            "sentiment_distribution": {
                "positive_percentage": round((sentiment_counts["positive"] / total_messages) * 100, 2),
                "negative_percentage": round((sentiment_counts["negative"] / total_messages) * 100, 2),
                "neutral_percentage": round((sentiment_counts["neutral"] / total_messages) * 100, 2)
            },
            "alert_distribution": alert_counts,
            "high_priority_count": alert_counts["high"],
            "average_confidence": round(sum([r.confidence_score for r in results]) / total_messages, 2),
            
            # Enhanced business intelligence summary
            "business_intelligence": {
                "average_churn_risk": round(total_churn_risk / total_messages, 1),
                "average_purchase_intent": round(total_purchase_intent / total_messages, 1),
                "high_value_customers": high_value_customers,
                "high_value_percentage": round((high_value_customers / total_messages) * 100, 1),
                "total_cost_savings": f"₹{cost_optimization_savings:.2f}",
                "customers_needing_retention": len([r for r in results if r.retention_action != "none"]),
                "immediate_response_required": len([r for r in results if "immediate" in r.response_prediction.best_response_time]),
                "marketing_opportunities": len([r for r in results if r.template_recommendation.primary_category == "marketing"]),
                "service_required": len([r for r in results if r.template_recommendation.primary_category == "service"])
            }
        }
        
        return {"results": results, "summary": summary}
    
    def _create_fallback_response(self, message, customer_id, agent_id, timestamp, error):
        """
        Create a safe fallback response when AI analysis fails
        """
        return SentimentResponse(
            message=message,
            sentiment=SentimentType.NEUTRAL,
            confidence_score=0.0,
            reasoning=f"Analysis error: {error}",
            alert_level="medium",
            
            # Conservative business intelligence values
            churn_probability=50,
            revenue_risk="at_risk",
            purchase_intent=25,
            customer_value_tier="medium_value",
            retention_action="follow_up",
            
            # Safe prediction defaults
            cost_prediction=CostPrediction(
                optimal_conversation_type="service",
                predicted_cost="₹0.00",
                cost_saved="₹0.00",
                reasoning="Fallback to safe service category due to analysis error"
            ),
            
            response_prediction=ResponsePrediction(
                success_probability=50,
                best_response_time="within_1_hour",
                escalation_probability=30,
                resolution_likelihood="medium"
            ),
            
            template_recommendation=TemplateRecommendation(
                primary_category="service",
                confidence=60,
                cost_impact="₹0.00",
                avoid_categories=["marketing"],
                reasoning="Safe fallback recommendation - use service category"
            ),
            
            customer_id=customer_id,
            agent_id=agent_id,
            timestamp=timestamp
        )