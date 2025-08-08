# 🎭 Advanced Sentiment 
!pip install keybert nrclex transformers spacy nltk matplotlib --quiet
!python -m spacy download xx_ent_wiki_sm --quiet

import math
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from keybert import KeyBERT
from nrclex import NRCLex
from transformers import pipeline, AutoTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from typing import List, Dict, Any, Optional
import nltk
nltk.download('vader_lexicon', quiet=True)

class SentimentEvolutionTracker:
    """
    Advanced sentiment analysis with temporal and contextual understanding.
    """

    def __init__(self):
        # --- Transformer sentiment (multilingual) with explicit tokenizer & truncation safety
        model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=self.tokenizer,
            device=-1  # CPU
        )

        # VADER for quick numeric polarity
        self.vader = SentimentIntensityAnalyzer()

        # Emotions (lexicon-based)
        self.emotion_extractor = NRCLex

        # Aspect/entity extraction (multilingual)
        self.nlp = spacy.load("xx_ent_wiki_sm")

        # Keyword extraction (multilingual)
        self.keybert = KeyBERT(model="distiluse-base-multilingual-cased-v1")

        # In-memory timeline
        self.sentiment_history: List[Dict[str, Any]] = []

    # Internals

    def _transformer_sentiment_chunked(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Runs the transformer sentiment on <=512-token chunks and averages per-label scores.
        Returns: {'label': top_label, 'score': top_avg, 'scores': {'positive': p, 'neutral': n, 'negative': q}}
        """
        if not text or not str(text).strip():
            return {"label": "neutral", "score": 0.0, "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0}}

        enc = self.tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) <= max_length:
            # ask for all scores so we can average consistently
            out = self.sentiment_model(text, truncation=True, max_length=max_length, return_all_scores=True)[0]
        else:
            # chunk by tokens
            chunks = []
            for i in range(0, len(ids), max_length):
                chunk_ids = ids[i:i+max_length]
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                chunks.append(chunk_text)
            outs = self.sentiment_model(chunks, truncation=True, max_length=max_length, return_all_scores=True)
            # average label scores across chunks
            sums = {}
            for chunk_scores in outs:
                for ls in chunk_scores:
                    sums.setdefault(ls["label"], []).append(ls["score"])
            out = [{"label": k, "score": sum(v)/len(v)} for k, v in sums.items()]

        # pick top label, and keep per-label map
        scores_map = {d["label"].lower(): float(d["score"]) for d in out}
        top_label = max(scores_map, key=scores_map.get)
        return {"label": top_label, "score": scores_map[top_label], "scores": scores_map}

    # ---------- Public API

    def analyze_sentiment(self, article_text: str, timestamp: Optional[Any] = None) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis.

        Returns dict with:
        - transformer_sentiment: {label, score, scores{pos/neu/neg}}
        - vader_scores: {compound, pos, neu, neg, sentiment_label, confidence}
        - emotions: list[(emotion, score)]
        - aspects: list[str]
        - keywords: list[str]
        """
        timestamp = timestamp or datetime.now()

        if not article_text or str(article_text).strip() == "":
            record = {
                "timestamp": pd.to_datetime(timestamp),
                "text": article_text,
                "transformer_sentiment": {"label": "neutral", "score": 0.0, "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0}},
                "vader_scores": {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0, 'sentiment_label': 'neutral', 'confidence': 0.0},
                "emotions": [],
                "aspects": [],
                "keywords": []
            }
            self.sentiment_history.append(record)
            return record

        # Transformer (chunk-safe)
        transformer_result = self._transformer_sentiment_chunked(article_text, max_length=512)

        # VADER
        vs = self.vader.polarity_scores(article_text)
        if vs['compound'] >= 0.05:
            vs['sentiment_label'] = 'positive'
        elif vs['compound'] <= -0.05:
            vs['sentiment_label'] = 'negative'
        else:
            vs['sentiment_label'] = 'neutral'
        vs['confidence'] = abs(vs['compound'])

        # Emotions
        emotions = self.emotion_extractor(article_text).top_emotions

        # Aspects/entities
        aspects = [ent.text for ent in self.nlp(article_text).ents]

        # Keywords
        keywords = [kw[0] for kw in self.keybert.extract_keywords(article_text, top_n=5)]

        record = {
            "timestamp": pd.to_datetime(timestamp),
            "text": article_text,
            "transformer_sentiment": transformer_result,
            "vader_scores": vs,
            "emotions": emotions,
            "aspects": aspects,
            "keywords": keywords
        }
        self.sentiment_history.append(record)
        return record

    def track_sentiment_over_time(self, articles_with_dates: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze sentiment trends over time.
        Expects: [{'article_text': '...', 'date': 'YYYY-MM-DD' or datetime}, ...]
        Returns a DataFrame with at least ['date','compound','sentiment_label'] and plots daily average.
        """
        sentiment_data = []
        for item in articles_with_dates:
            text = item.get('article_text', '')
            ts = item.get('date', None)
            res = self.analyze_sentiment(text, timestamp=ts)
            sentiment_data.append({
                "date": pd.to_datetime(res["timestamp"]),
                "compound": res["vader_scores"]["compound"],
                "sentiment_label": res["vader_scores"]["sentiment_label"]
            })

        df = pd.DataFrame(sentiment_data).sort_values("date")
        if df.empty:
            print("No data available.")
            return df

        daily = df.set_index("date")["compound"].resample("D").mean().fillna(0)

        plt.figure(figsize=(12, 6))
        daily.plot()
        plt.title("Daily Average Sentiment Over Time")
        plt.xlabel("Date")
        plt.ylabel("Average VADER Compound")
        plt.grid(True)
        plt.show()

        return df

    def detect_sentiment_anomalies(self, sentiment_timeline: pd.DataFrame, window_size: int = 7, threshold: float = 2.0) -> Optional[pd.DataFrame]:
        """
        Identify unusual sentiment patterns in a timeline DataFrame with columns ['date','compound'].
        Uses deviation from rolling mean > threshold*rolling_std.
        Returns a DataFrame of anomalies or None.
        """
        if sentiment_timeline is None or sentiment_timeline.empty:
            print("No sentiment timeline data provided for anomaly detection.")
            return None
        if "date" not in sentiment_timeline.columns or "compound" not in sentiment_timeline.columns:
            raise ValueError("sentiment_timeline must have 'date' and 'compound' columns.")

        df = sentiment_timeline.copy()
        df = df.sort_values("date").reset_index(drop=True)
        df["rolling_mean"] = df["compound"].rolling(window=window_size, min_periods=window_size).mean()
        df["rolling_std"] = df["compound"].rolling(window=window_size, min_periods=window_size).std()
        df["deviation"] = (df["compound"] - df["rolling_mean"]).abs()
        df["threshold_val"] = threshold * df["rolling_std"]

        anomalies = df[df["deviation"] > df["threshold_val"]].dropna().copy()
        if anomalies.empty:
            print("No significant sentiment anomalies detected.")
            return None

        # Plot with anomalies highlighted
        plt.figure(figsize=(12, 6))
        plt.plot(df["date"], df["compound"], label="Sentiment")
        if not anomalies.empty:
            plt.scatter(anomalies["date"], anomalies["compound"], label="Anomaly", zorder=5)
        plt.title("Sentiment Anomalies Over Time")
        plt.xlabel("Date")
        plt.ylabel("VADER Compound")
        plt.legend()
        plt.grid(True)
        plt.show()

        return anomalies
      
    def table_of_scores(
        self,
        df: pd.DataFrame,
        text_col: str = "full_text",
        include_details: bool = True,
        preview: int = 20,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Build a detailed sentiment table from a DataFrame column.
        - df: pandas DataFrame containing text
        - text_col: column name with raw text (default 'full_text')
        - include_details: include emotions/aspects/keywords columns
        - preview: number of rows to display (head)
        - plot: plot VADER compound by row index

        Returns: pandas DataFrame with scores and details.
        """
        from IPython.display import display

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not found in DataFrame")

        rows = []
        for i, t in enumerate(df[text_col].fillna("").astype(str).tolist(), start=1):
            try:
                res = self.analyze_sentiment(t)

                tr = res.get("transformer_sentiment") or {}
                vs = res.get("vader_scores") or {}

                if include_details:
                    emotions_list = res.get("emotions") or []   # [(emo, score), ...]
                    aspects_list = res.get("aspects") or []      # ["entity", ...]
                    keywords_list = res.get("keywords") or []    # ["kw", ...]

                    emotions_str = ", ".join(f"{e}:{round(s,3)}" for e, s in emotions_list)
                    aspects_str = ", ".join(aspects_list)
                    keywords_str = ", ".join(keywords_list)
                else:
                    emotions_list = aspects_list = keywords_list = []
                    emotions_str = aspects_str = keywords_str = ""

                row = {
                    "row": i,
                    "text_preview": (t[:160] + "…") if len(t) > 160 else t,
                    "full_text": t,
                    # transformer
                    "transformer_label": tr.get("label"),
                    "transformer_score": tr.get("score"),
                    "transformer_pos": (tr.get("scores") or {}).get("positive"),
                    "transformer_neu": (tr.get("scores") or {}).get("neutral"),
                    "transformer_neg": (tr.get("scores") or {}).get("negative"),
                    # vader
                    "vader_compound": vs.get("compound"),
                    "vader_pos": vs.get("pos"),
                    "vader_neu": vs.get("neu"),
                    "vader_neg": vs.get("neg"),
                    "sentiment_label": vs.get("sentiment_label"),
                    "confidence": vs.get("confidence"),
                }

                if include_details:
                    row.update({
                        "emotions_top_display": emotions_str,
                        "emotions_top_raw": emotions_list,
                        "aspects_display": aspects_str,
                        "aspects_raw": aspects_list,
                        "keywords_display": keywords_str,
                        "keywords_raw": keywords_list,
                    })

                rows.append(row)

            except Exception as e:
                rows.append({
                    "row": i,
                    "text_preview": (t[:160] + "…") if len(t) > 160 else t,
                    "full_text": t,
                    "error": str(e),
                })

        scores_df = pd.DataFrame(rows)

        # Display preview
        display(scores_df.head(preview))

        # Optional plot
        if plot and "vader_compound" in scores_df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(scores_df["row"], scores_df["vader_compound"])
            plt.title("Sentiment by Row (VADER compound)")
            plt.xlabel("Row")
            plt.ylabel("Compound Score")
            plt.grid(True)
            plt.show()

        return scores_df



