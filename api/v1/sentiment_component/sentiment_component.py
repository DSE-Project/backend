import praw
import pandas as pd
import numpy as np
from datetime import datetime
import re
from textblob import TextBlob
from collections import Counter
from typing import List, Dict, Tuple

# Import your LLM wrapper
from .langchain_helper import SentimentAnalysisLLM  # remove relative import for direct execution


class RedditRecessionSentimentAgent:
    def __init__(self):
        self.reddit = self.setup_reddit()

    def setup_reddit(self):
        """Initialize Reddit client with credentials"""
        return praw.Reddit(
            client_id="yUBDEQfPyjGLgwswf7YViw",
            client_secret="J-5ErmNyROkgO5mFyGb_-PjO3LkhdA",
            user_agent="USRecessionSentiment/1.0 by u/Dear-Average7458",
            username="Dear-Average7458",
            password="Navas02$"
        )

    # --- Scraping methods ---
    def scrape_reddit_economic_data(self, days_back: int = 7, max_posts_per_sub: int = 20) -> Tuple[List[Dict], List[Dict]]:
        posts_data = []
        comments_data = []

        economic_subreddits = ['economics', 'economy', 'stocks', 'investing', 'personalfinance']
        recession_keywords = ['recession', 'inflation', 'GDP', 'unemployment', 'market', 'fed']

        for subreddit_name in economic_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                search_query = " OR ".join(recession_keywords)

                for post in subreddit.search(search_query, limit=2, sort='new'):
                    self._process_post(post, posts_data, comments_data, days_back)
                for post in subreddit.hot(limit=2):
                    self._process_post(post, posts_data, comments_data, days_back)
                if subreddit_name in ['economics', 'economy', 'stocks']:
                    for post in subreddit.new(limit=2):
                        self._process_post(post, posts_data, comments_data, days_back)

            except Exception as e:
                print(f"⚠️ Error with r/{subreddit_name}: {str(e)[:100]}...")
                continue

        return posts_data, comments_data

    def _process_post(self, post, posts_data: List[Dict], comments_data: List[Dict], days_back: int):
        try:
            post_date = datetime.fromtimestamp(post.created_utc)
            if (datetime.now() - post_date).days > days_back:
                return

            content = (post.title + " " + (post.selftext or "")).lower()
            economic_keywords = ['recession', 'economy', 'inflation', 'rates', 'fed', 'gdp', 'unemployment', 'market']
            if not any(keyword in content for keyword in economic_keywords):
                return

            post_data = {
                'source': 'reddit',
                'subreddit': post.subreddit.display_name,
                'title': post.title,
                'content': post.selftext or "",
                'author': str(post.author) if post.author else 'deleted',
                'upvotes': post.score,
                'comments_count': post.num_comments,
                'date': post_date,
                'url': f"https://reddit.com{post.permalink}",
                'sentiment': self.analyze_sentiment(post.title + " " + (post.selftext or "")),
                'economic_indicators': self.extract_economic_indicators(post.title + " " + (post.selftext or ""))
            }
            posts_data.append(post_data)

            # Process comments
            try:
                post.comments.replace_more(limit=3)
                for comment in post.comments.list()[:15]:
                    if hasattr(comment, 'body') and len(comment.body) > 20 and not comment.stickied and comment.score > 0:
                        comment_data = {
                            'source': 'reddit_comment',
                            'subreddit': post.subreddit.display_name,
                            'post_title': post.title,
                            'content': comment.body,
                            'author': str(comment.author) if comment.author else 'deleted',
                            'upvotes': comment.score,
                            'date': datetime.fromtimestamp(comment.created_utc),
                            'sentiment': self.analyze_sentiment(comment.body),
                            'economic_indicators': self.extract_economic_indicators(comment.body),
                            'post_url': f"https://reddit.com{post.permalink}"
                        }
                        comments_data.append(comment_data)
            except:
                pass

        except:
            pass

    # --- Analysis methods ---
    def analyze_sentiment(self, text: str) -> str:
        try:
            if not text or len(text.strip()) < 10:
                return 'neutral'
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.15:
                return 'positive'
            elif polarity < -0.15:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def extract_economic_indicators(self, text: str) -> List[str]:
        indicators = {
            'recession': r'recession|economic downturn|bear market|market crash',
            'inflation': r'inflation|price increase|CPI|consumer price',
            'interest_rates': r'interest rates|fed rates|rate hike|rate cut',
            'employment': r'unemployment|jobs|hiring|layoff|job market',
            'growth': r'GDP|economic growth|recovery|expansion'
        }
        found = [k for k, v in indicators.items() if re.search(v, text.lower(), re.IGNORECASE)]
        return found

    # --- Dashboard & LLM integration ---
    def run_dashboard_analysis(self, days_back: int = 5, max_posts_per_sub: int = 15):
        """Run analysis and return structured data for frontend including LLM summary"""
        posts, comments = self.scrape_reddit_economic_data(days_back, max_posts_per_sub)
        report = f"Collected {len(posts)} posts and {len(comments)} comments."

        # --- LLM safe summary ---
        llm = SentimentAnalysisLLM()

        # Reduce text sent to LLM: only titles + short content snippet
        def prepare_for_llm(data_list, key_title='title', key_content='content', snippet_len=50):
            return [
                {key_title: item[key_title], key_content: item[key_content][:snippet_len]}
                for item in data_list
            ]

        llm_input = {
            "posts": prepare_for_llm(posts),
            "comments": prepare_for_llm(comments, key_title='post_title', key_content='content')
        }

        # Generate summary safely
        try:
            llm_summary = llm.generate_summary(llm_input)
        except Exception as e:
            llm_summary = f"Error generating LLM summary: {str(e)}"

        return {
            "summary": {
                "total_posts": len(posts),
                "total_comments": len(comments),
                "total_points": len(posts) + len(comments),
                "time_range_days": self._get_data_time_range(posts + comments),
                "avg_engagement": round(self._get_avg_engagement(posts + comments), 2),
            },
            "report": report,
            "llm_summary": llm_summary,
            "posts": posts,
            "comments": comments
        }

    # --- Helpers ---
    def _get_data_time_range(self, all_data):
        if not all_data:
            return 0
        dates = [item['date'] for item in all_data if 'date' in item]
        return (datetime.now() - min(dates)).days if dates else 0

    def _get_avg_engagement(self, all_data):
        if not all_data:
            return 0
        upvotes = [item.get('upvotes', 0) for item in all_data]
        return np.mean(upvotes) if upvotes else 0


# --- Run analysis ---
if __name__ == "__main__":
    agent = RedditRecessionSentimentAgent()
    result = agent.run_dashboard_analysis()
    print("=== Dashboard LLM Summary ===")
    print(result["llm_summary"])
