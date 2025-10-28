# AI Course Copilot – Phase 03
Reproduces our training and evaluation (Weeks 7–10).

**Colab (view-only):** https://colab.research.google.com/drive/1wi_w2jAf26VI0QyrWZP9movX6DGQM6g9#scrollTo=6fGZbgzXz-Uc

## Run on Colab
```bash
!git clone https://github.com/oscarecortez361/ai-course-copilot-phase03
%cd ai-course-copilot-phase03
!pip install -r requirements.txt
!python scripts/01_ingest.py && python scripts/02_clean_chunk.py && python scripts/03_embed_index.py
!python scripts/train_retrieval.py --config config.yaml
!python scripts/eval_retrieval.py --config config.yaml --out eval/results/metrics.json
