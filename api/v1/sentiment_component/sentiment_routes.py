# sentiment_routes.py
from fastapi import APIRouter
from api.v1.sentiment_component.sentiment_component import RedditRecessionSentimentAgent

router = APIRouter()

@router.get("/reddit-sentiment")
def get_reddit_sentiment():
    agent = RedditRecessionSentimentAgent()
    result = agent.run_dashboard_analysis()
    return result
