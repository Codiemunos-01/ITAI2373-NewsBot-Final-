import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import networkx as nx # Import networkx for plotting graph


class TopicDiscoveryEngine:
    """
    Advanced topic modeling for discovering themes and trends.
    Supports 'lda' (probabilistic) and 'nmf' (parts-based).
    """

    def __init__(
        self,
        n_topics: int = 10,
        method: str = "lda",
        max_features: int = 30000,
        min_df: Union[int, float] = 2, # Adjusted min_df
        max_df: Union[int, float] = 0.95, # Adjusted max_df
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42
    ):
        assert method in {"lda", "nmf"}, "method must be 'lda' or 'nmf'"
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.random_state = random_state

        # vectorizer choice depends on method
        if self.method == "lda":
            self.vectorizer = CountVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range
            )
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=random_state,
                learning_method="batch"
            )
        else:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range
            )
            self.model = NMF(
                n_components=n_topics,
                init="nndsvd",
                random_state=random_state,
                max_iter=400
            )

        # Fitted artifacts
        self._fitted = False
        self.doc_term: Optional[csr_matrix] = None
        self.doc_topic: Optional[np.ndarray] = None
        self.topic_word: Optional[np.ndarray] = None
        self.feature_names: Optional[np.ndarray] = None
        self.topic_top_terms: Optional[List[List[str]]] = None
        self.topic_labels: Optional[List[str]] = None
        self.doc_dates: Optional[pd.Series] = None

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _basic_clean(text: str) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # URLs
        text = re.sub(r"[^a-z0-9\s]", " ", text)                # punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _label_topics(self, top_n: int = 6) -> List[str]:
        labels = []
        for words in self.topic_top_terms:
            labels.append(", ".join(words[:top_n]))
        return labels

    def _top_words(self, n: int = 15) -> List[List[str]]:
        top_words = []
        for k in range(self.n_topics):
            # topic_word: topics x terms
            row = self.topic_word[k, :]
            idx = np.argsort(row)[::-1][:n]
            top_words.append([self.feature_names[i] for i in idx])
        return top_words

    def _topic_quality(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Returns:
          - topic_diversity: unique words / (topics * top_k)
          - coherence_umass_proxy: average log co-occurrence score over top word pairs
        """
        top_terms = self._top_words(n=top_k)

        # Diversity
        flat = [w for topic in top_terms for w in topic]
        diversity = len(set(flat)) / (len(top_terms) * top_k + 1e-9)

        # Coherence proxy (UMass-like using doc-term binary co-occurrence)
        if self.doc_term is None:
            coherence = np.nan
        else:
            # binary presence
            Xbin = self.doc_term.copy()
            Xbin.data = np.ones_like(Xbin.data)
            # term frequencies
            tf = np.asarray(Xbin.sum(axis=0)).ravel() + 1e-12
            # precompute for speed
            coherence_scores = []
            for topic in top_terms:
                # indices of top terms
                idxs = [np.where(self.feature_names == w)[0][0] for w in topic if w in self.feature_names]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        wi, wj = idxs[i], idxs[j]
                        # co-occurrence count
                        co = Xbin[:, wi].multiply(Xbin[:, wj]).sum() + 1e-12
                        coherence_scores.append(math.log(co / tf[wj]))
            coherence = float(np.mean(coherence_scores)) if coherence_scores else np.nan

        return {"topic_diversity": diversity, "coherence_umass_proxy": coherence}

    # -------------------------
    # Public API
    # -------------------------
    def fit_topics(
        self,
        documents: List[str],
        dates: Optional[Union[List[Any], pd.Series]] = None,
        top_words: int = 15
    ):
        """
        Fit the topic model on a collection of documents.
        Optionally pass dates (same length) to enable trend tracking later.
        """
        assert isinstance(documents, (list, pd.Series)), "documents must be a list/Series of strings"
        docs_clean = [self._basic_clean(x) for x in documents]

        # Vectorize
        X = self.vectorizer.fit_transform(docs_clean)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Fit model
        self.model.fit(X)

        # Topic-word matrix
        if self.method == "lda":
            self.topic_word = self.model.components_  # already counts-ish
            # Doc-topic (gamma): transform gives topic distribution
            self.doc_topic = self.model.transform(X)  # shape: n_docs x n_topics
            # Normalize doc-topic to sum to 1
            self.doc_topic = normalize(self.doc_topic, norm="l1", axis=1)
        else:
            self.topic_word = self.model.components_  # NMF weights
            self.doc_topic = self.model.transform(X)
            self.doc_topic = normalize(self.doc_topic, norm="l1", axis=1)

        self.doc_term = X.tocsr()
        self.topic_top_terms = self._top_words(n=top_words)
        self.topic_labels = self._label_topics()
        self._fitted = True

        # Keep dates if provided
        if dates is not None:
            self.doc_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
        else:
            self.doc_dates = None

        return {
            "topic_labels": self.topic_labels,
            "quality": self._topic_quality(top_k=min(10, top_words))
        }

    def get_article_topics(self, article_text: str, top_n: int = 5) -> List[Tuple[int, float, str]]:
        """
        Return top topics (index, weight, label) for a single article.
        """
        assert self._fitted, "Call fit_topics() first."
        clean = self._basic_clean(article_text)
        vec = self.vectorizer.transform([clean])
        dist = self.model.transform(vec)
        dist = normalize(dist, norm="l1", axis=1)[0]
        idx = np.argsort(dist)[::-1][:top_n]
        return [(int(i), float(dist[i]), self.topic_labels[i]) for i in idx]

    def track_topic_trends(
        self,
        documents: Optional[List[str]] = None,
        dates: Optional[Union[List[Any], pd.Series]] = None,
        freq: str = "W"
    ) -> pd.DataFrame:
        """
        Assign each doc to its most probable topic and aggregate over time.
        If documents are provided, they will be fitted/transformed with existing vectorizer/model.
        If not provided, use the last fitted doc-topic and stored dates.
        Returns a pivot with counts per topic over time index.
        """
        assert self._fitted, "Call fit_topics() first."

        if documents is not None:
            docs_clean = [self._basic_clean(x) for x in documents]
            X = self.vectorizer.transform(docs_clean)
            doc_topic = self.model.transform(X)
            doc_topic = normalize(doc_topic, norm="l1", axis=1)
            ts = pd.to_datetime(pd.Series(dates), errors="coerce") if dates is not None else None
        else:
            doc_topic = self.doc_topic
            ts = self.doc_dates

        if ts is None:
            raise ValueError("Dates are required to track topic trends. Pass `dates` or fit with dates first.")

        # Most likely topic per doc
        top_idx = np.argmax(doc_topic, axis=1)
        df = pd.DataFrame({"date": ts, "topic": top_idx}).dropna()
        df["date"] = pd.to_datetime(df["date"])
        df["count"] = 1

        trend = (
            df.set_index("date")
              .groupby("topic")
              .resample(freq)["count"]
              .sum()
              .unstack(0)
              .fillna(0)
        )
        # Add readable column names
        trend.columns = [f"Topic {i}: {self.topic_labels[i]}" for i in trend.columns]

        # Plot
        plt.figure(figsize=(12, 6))
        trend.plot(ax=plt.gca())
        plt.title(f"Topic Trends Over Time ({freq} frequency)")
        plt.xlabel("Date")
        plt.ylabel("Document Count")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()

        return trend

    def visualize_topics(self, top_n: int = 12) -> pd.DataFrame:
        """
        Return a DataFrame of topic -> top words, and print nicely.
        """
        assert self._fitted, "Call fit_topics() first."
        topic_words = self._top_words(n=top_n)
        df = pd.DataFrame({
            "topic": [f"Topic {i}" for i in range(self.n_topics)],
            "label": self._label_topics(top_n=6),
            "top_words": [", ".join(ws) for ws in topic_words],
        })
        # Pretty print
        for i, row in df.iterrows():
            print(f"{row['topic']} — {row['label']}")
            print(f"  {row['top_words']}\n")
        return df
