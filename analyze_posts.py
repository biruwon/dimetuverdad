from transformers import pipeline
import re
import torch
from retrieval import retrieve_evidence_for_post, format_evidence
import sqlite3
import json
import os
import argparse

# DB path (same as other scripts)
DB_PATH = os.path.join(os.path.dirname(__file__), 'accounts.db')


def init_analysis_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_url TEXT,
        analysis_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()


def save_analysis(tweet_url: str, analysis_obj: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO analyses (tweet_url, analysis_json) VALUES (?, ?)', (tweet_url, json.dumps(analysis_obj, ensure_ascii=False)))
    conn.commit()
    conn.close()


# Far-right motif list (common themes and codewords used by activists)
FAR_RIGHT_MOTIFS = [
    'inmigr', 'inmigrantes', 'migración', 'reemplaz', 'soros', 'globalist', 'deep state', 'islamiz',
    'ilegales', 'okupa', 'terror', 'nacional', 'identidad', 'familia', 'tradición', 'invasión', 'traidor',
    'raza', 'suprem', 'ilegales', 'qn', 'roedores', 'plandemia', 'mascara', 'vacuna', 'microchip', 'genocid',
    'degener', 'sodom', 'antiespa', 'invasor', 'traicion'
]


def far_right_score(text: str) -> float:
    t = text.lower()
    score = 0
    for m in FAR_RIGHT_MOTIFS:
        if m in t:
            score += 1
    # calls-to-action verbs or all-caps words add weight
    if re.search(r"\b(venid[ae]|salid|luchar|tomar|recuperar|ocupar|atacar|salid|marchad|concentrad|revolución)\b", t):
        score += 2
    if re.search(r"\b[A-Z]{3,}\b", text):
        score += 1
    # scale score to 0..1 where higher means stronger far-right signal
    max_possible = max(1, len(FAR_RIGHT_MOTIFS) + 3)
    return min(1.0, score / max_possible)


# init_analysis_table() will be called later only if saving is enabled;
# this avoids DB I/O when running in fast/analysis-only mode.

# Zero-shot classification pipeline (multilingual-capable)
topic_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Lazy LLM loader: will instantiate llm_pipe on first use to avoid heavy startup cost
llm_pipe = None


def load_llm():
    """Instantiate the LLM pipeline on first use and return it. Safe and idempotent."""
    global llm_pipe
    if llm_pipe is not None:
        return llm_pipe
    try:
        if torch.cuda.is_available():
            llm_pipe = pipeline(
                "text-generation",
                model="google/gemma-3n-e2b-it",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            llm_pipe = pipeline(
                "text-generation",
                model="google/gemma-3n-e2b-it",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            llm_pipe = pipeline(
                "text-generation",
                model="google/gemma-3n-e2b-it",
                device=-1,
                trust_remote_code=True,
            )
        return llm_pipe
    except Exception as e:
        print(f"Warning: could not load Gemma model locally: {e}")
        llm_pipe = None
        return None


# Labels for topic/politics detection (expanded for extremist/activist signals)
politics_labels = [
    "política", "gobierno", "elecciones", "política pública", "ley", "economía", "sanidad", "seguridad",
    "extrema derecha", "extrema izquierda", "activismo", "protesta", "incitación", "discurso de odio"
]
# Labels for misinformation/factual/opinion
misinfo_labels = ["desinformación", "factual", "opinión", "engañoso", "rumor", "engaño"]

# Lightweight heuristic to spot factual claims or red-flags often associated with misinformation
claim_patterns = [
    r"\b\d+[\.,]?\d*\b",            # numbers
    r"\bpor ciento\b",                # percentage
    r"\bmillones?\b",                 # millions
    r"\bvacun[ae]\b",                # vaccine
    r"\bmicrochip\b",                # microchip
    r"\bprohibi[rsn]\b",             # prohibir/prohiben
    r"\bmult[ao]n\b",                # multan/multa
    r"\binvestig[ai]r\b",            # investigar/investigan
]
claim_regex = re.compile("|".join(claim_patterns), re.IGNORECASE | re.UNICODE)


def looks_like_claim(text: str) -> bool:
    """Return True if the text contains patterns commonly found in concrete factual claims."""
    return bool(claim_regex.search(text))


def main(posts_list, skip_retrieval=False, skip_save=False):
    # initialize DB only if saving enabled
    if not skip_save:
        init_analysis_table()

    # collector for analysis outputs
    results = []

    # Fast stage: parallel zero-shot + heuristics to select candidates
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fast_check(text):
        topic_result = topic_pipe(text, candidate_labels=politics_labels, hypothesis_template="Este texto trata sobre {}.")
        misinfo_result = topic_pipe(text, candidate_labels=misinfo_labels, hypothesis_template="Este texto es {}.")
        best_topic_score = topic_result['scores'][0]
        misinfo_dict = dict(zip(misinfo_result['labels'], misinfo_result['scores']))
        misinfo_score = misinfo_dict.get('desinformación', 0.0)
        claim_flag = looks_like_claim(text)
        return (text, best_topic_score, misinfo_score, claim_flag)

    candidates = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fast_check, t): t for t in posts_list}
        for fut in as_completed(futures):
            t, best_topic_score, misinfo_score, claim_flag = fut.result()
            political_threshold = 0.45
            misinfo_threshold = 0.30
            is_political = best_topic_score >= political_threshold
            is_probably_misinfo = misinfo_score >= misinfo_threshold
            print(f"\nPost: {t}")
            print(f"Topic score: {best_topic_score:.2f}, misinfo: {misinfo_score:.2f}, claim_flag={claim_flag}")
            if is_political or is_probably_misinfo or claim_flag:
                candidates.append(t)

    # Sequential deep analysis on candidates (load LLM lazily)
    for text in candidates:
        print("-> Running deep analysis (Gemma) on candidate:", text[:80])
        evidence = []
        if not skip_retrieval:
            try:
                evidence = retrieve_evidence_for_post(text, max_per_source=2)
            except Exception as e:
                print(f"Evidence retrieval failed: {e}")

        prompt = (
            "Contexto: Estos posts pertenecen a actividad de la extrema derecha en España dirigida contra el Gobierno; ten en cuenta ese marco al analizar sesgos y posibles llamadas a la acción.\n\n"
            "Instrucciones: Responde SOLO en español. No traduzcas el texto. No inventes enlaces ni cites artículos externos. Devuelve SOLO un objeto JSON válido con las siguientes claves: \n"
            "- topic: cadena breve con el tema principal (ej: 'gobierno', 'protesta')\n"
            "- misinformation: 'yes' o 'no'\n"
            "- misinformation_reason: explicación corta si 'yes', cadena vacía si 'no'\n"
            "- political_bias: 'left' | 'right' | 'neutral' | 'unknown'\n"
            "- bias_confidence: número entre 0.0 y 1.0\n"
            "- calls_to_action: 'yes' o 'no'\n"
            "- targeted_group: grupo objetivo si aplica (ej: 'gobierno', 'inmigrantes'), o cadena vacía\n"
            "- explanation: explicación breve y clara del razonamiento que llevó a la decisión anterior\n"
            "- extra_info: un objeto con claves 'retrieval_hint' (palabras clave sugeridas) y 'evidence' (lista vacía que el sistema rellenará)\n\n"
            f"Post: {text}\n\n"
            "JSON:"
        )

        pipe = load_llm()
        if pipe is None:
            print("LLM not available locally; skipping deep analysis.")
            continue
        try:
            llm_result = pipe(prompt, max_new_tokens=256)
            if isinstance(llm_result, list) and llm_result:
                generated = llm_result[0].get("generated_text", "")
            elif isinstance(llm_result, dict):
                generated = llm_result.get("generated_text", "")
            else:
                generated = str(llm_result)
            generated = (generated or "").strip()
        except Exception as e:
            print(f"LLM generation failed: {e}")
            continue

        # parse and attach evidence similarly to previous logic
        import json as _json
        json_obj = None
        try:
            start = generated.find('{')
            end = generated.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw = generated[start:end+1]
                json_obj = _json.loads(raw)
        except Exception:
            json_obj = None

        if json_obj is None:
            print("Gemma analysis (raw, could not parse JSON):\n", generated)
            continue

        retrieval_hint = ''
        if isinstance(json_obj.get('extra_info'), dict):
            retrieval_hint = json_obj['extra_info'].get('retrieval_hint', '') or ''
        if not retrieval_hint:
            parts = []
            if json_obj.get('topic'):
                parts.append(str(json_obj['topic']))
            if json_obj.get('targeted_group'):
                parts.append(str(json_obj['targeted_group']))
            retrieval_hint = ' '.join(parts) or text[:200]

        ev = []
        if not skip_retrieval:
            try:
                ev = retrieve_evidence_for_post(retrieval_hint, max_per_source=3)
            except Exception as e:
                print(f"Retrieval failed: {e}")

        extra = json_obj.get('extra_info') if isinstance(json_obj.get('extra_info'), dict) else {}
        extra['evidence'] = ev
        json_obj['extra_info'] = extra
        json_obj['far_right_score'] = far_right_score(text)

        tweet_url = json_obj.get('tweet_url') if isinstance(json_obj.get('tweet_url'), str) else None
        if not tweet_url:
            tweet_url = text[:140]

        if not skip_save:
            try:
                save_analysis(tweet_url, json_obj)
            except Exception as e:
                print(f"Failed to save analysis to DB: {e}")

        # Append to results list for writing to file later
        results.append(json_obj)
        print("Gemma analysis (parsed JSON with retrieved evidence):\n", _json.dumps(json_obj, ensure_ascii=False, indent=2))

    # At end of run, write results to analysis_results.json (overwrite)
    try:
        out_path = os.path.join(os.path.dirname(__file__), 'analysis_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(results)} analysis results to {out_path}")
    except Exception as e:
        print(f"Failed to write analysis results file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-retrieval', action='store_true', help='Skip retrieval of external evidence')
    parser.add_argument('--skip-save', action='store_true', help='Skip saving analysis to the DB')
    parser.add_argument('--fast', action='store_true', help='Shortcut: skip retrieval and saving')
    parser.add_argument('--from-db', action='store_true', help='Load posts from DB instead of embedded defaults')
    parser.add_argument('--results-file', help='Optional path to write analysis results (defaults to analysis_results.json)')
    args = parser.parse_args()
    if args.fast:
        args.skip_retrieval = True
        args.skip_save = True

    posts_to_analyze = []
    if args.from_db:
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT content FROM tweets ORDER BY created_at DESC LIMIT 1000')
            rows = c.fetchall()
            conn.close()
            posts_to_analyze = [r[0] for r in rows if r and r[0].strip()]
            print(f"Loaded {len(posts_to_analyze)} posts from DB for analysis")
        except Exception as e:
            print(f"Failed to load posts from DB: {e}")

    if not posts_to_analyze:
        print("No posts to analyze. Run with --from-db to analyze saved tweets or call main() programmatically with a posts list.")
    else:
        main(posts_to_analyze, skip_retrieval=args.skip_retrieval, skip_save=args.skip_save)
