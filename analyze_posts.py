from transformers import pipeline
import re

# Zero-shot classification pipeline (multilingual-capable)
topic_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Use Gemma for deep analysis
llm_pipe = pipeline("text-generation", model="google/gemma-2-2b-it", device_map="auto")

# Example posts to analyze
posts = [
    "Prohíben limpiar y desbrozar el monte, y si lo haces te multan. Y ahora te van a investigar por no haberlo limpiado.",
    "Hoy hace sol en Madrid.",
    "🔴 #ÚLTIMAHORA | Pedro Sánchez cierra parte del espacio marítimo de Lanzarote hasta el 31 de agosto para su recreo vacacional en La Mareta. El Gobierno del pueblo.",
    "🔴 La “extrema derecha” esquivando y toreando las difamaciones, insultos y mentiras de los zurdos, óleo sobre lienzo"
    "Ocupan el piso de una mujer de 90 años, heredado de su hijo fallecido, y al acomplejado de Agente Anacleto no se le ocurre otra cosa que justificarlo. Es repugnante tener que aguantar a estos mercenarios sin decencia alguna, haciendo el papel de siervos del Gobierno."
]

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


for text in posts:
    print(f"\nPost: {text}")
    # 1. Separate zero-shot checks for politics and misinformation
    topic_result = topic_pipe(text, candidate_labels=politics_labels, hypothesis_template="Este texto trata sobre {}.")
    misinfo_result = topic_pipe(text, candidate_labels=misinfo_labels, hypothesis_template="Este texto es {}.")

    # pick best topic match
    best_topic_label = topic_result['labels'][0]
    best_topic_score = topic_result['scores'][0]
    print(f"Topic detection: {best_topic_label} (score: {best_topic_score:.2f})")

    # misinfo scores
    misinfo_dict = dict(zip(misinfo_result['labels'], misinfo_result['scores']))
    misinfo_score = misinfo_dict.get('desinformación', 0.0)
    factual_score = misinfo_dict.get('factual', 0.0)
    opinion_score = misinfo_dict.get('opinión', 0.0)
    print(f"Misinfo probs: desinformación={misinfo_score:.2f}, factual={factual_score:.2f}, opinión={opinion_score:.2f}")

    # 2. Heuristic claim detection
    claim_flag = looks_like_claim(text)
    print(f"Claim-like content detected: {claim_flag}")

    # 3. Combined decision: if topic is political OR misinfo likelihood or claim present -> run LLM
    political_threshold = 0.45
    misinfo_threshold = 0.30

    is_political = best_topic_score >= political_threshold
    is_probably_misinfo = misinfo_score >= misinfo_threshold

    if is_political or is_probably_misinfo or claim_flag:
        print("-> Running deep analysis (Gemma)...")
        # Enhanced prompt: context-aware (far-right activity), strict Spanish output, JSON schema, no links, no translation
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
            "- notes: observaciones breves\n\n"
            f"Post: {text}\n\n"
            "JSON:"
        )
        llm_result = llm_pipe(prompt, max_new_tokens=256)
        generated = llm_result[0]["generated_text"].strip()
        # Try to extract JSON from generated text: print raw for now
        print("Gemma analysis (raw):\n", generated)
    else:
        print("No political/misinfo content detected. Skipping deep analysis.")
