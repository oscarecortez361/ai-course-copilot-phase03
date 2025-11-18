import os, yaml, pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class Backend:
    def __init__(self, settings_path="app/settings.yaml"):
        self.cfg = yaml.safe_load(open(settings_path, "r", encoding="utf-8"))
        p = self.cfg["paths"]
        # corpus + ids
        self.chunks = pd.read_parquet(p["chunks_parquet"])
        self.index = faiss.read_index(p["faiss_index"])
        self.ids = pd.read_csv(p["ids_csv"])["chunk_id"].tolist()
        # embedder
        self.model = SentenceTransformer(self.cfg["retrieval"]["embed_model"])
        # feedback store
        self.feedback_csv = self.cfg.get("feedback_csv", "data/feedback.csv")
        os.makedirs(os.path.dirname(self.feedback_csv), exist_ok=True)

    def _embed(self, texts):
        vecs = self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(vecs)  # cosine
        return vecs

    def retrieve(self, query, k=None):
        k = k or self.cfg["retrieval"]["k"]
        qv = self._embed([query])
        D, I = self.index.search(qv, k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            cid = self.ids[idx]
            row = self.chunks.loc[self.chunks["chunk_id"] == cid].iloc[0]
            out.append({
                "chunk_id": cid,
                "score": float(score),
                "text": row["text"],
                "doc_title": row.get("doc_title", ""),
                "doc_id": row.get("doc_id", "")
            })
        return out

    def answer_extractive(self, query, passages, max_chars=None):
        """Simple extractive summary from top passages (no external LLM)."""
        max_chars = max_chars or self.cfg["retrieval"]["max_context_chars"]
        ctx, used = "", []
        for p in passages:
            frag = p["text"].strip().replace("\n", " ")
            if not frag:
                continue
            if len(ctx) + 1 + len(frag) > max_chars:
                break
            ctx = (ctx + " " + frag).strip()
            used.append(p)
        if not used:
            return "I couldn't find relevant passages for that question.", []
        intro = "Here’s what the course docs say: "
        return (intro + ctx)[:max_chars], [u["chunk_id"] for u in used[:3]]

    def record_feedback(self, query, answer, helpful, comment=""):
        row = {
            "query": query,
            "helpful": helpful,
            "comment": comment,
            "answer_preview": (answer[:200] + "…")
        }
        mode = "a" if os.path.exists(self.feedback_csv) else "w"
        header = not os.path.exists(self.feedback_csv)
        pd.DataFrame([row]).to_csv(self.feedback_csv, mode=mode, index=False, header=header)
