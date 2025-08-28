"""Run analysis on embedded default posts (no DB) in fast mode.
This script uses the enhanced analyzer to process default posts.
"""
from enhanced_analyzer import EnhancedAnalyzer

# Default/example posts kept separate from the analysis module
posts = [
    "Prohíben limpiar y desbrozar el monte, y si lo haces te multan. Y ahora te van a investigar por no haberlo limpiado.",
    "Hoy hace sol en Madrid.",
    "🔴 #ÚLTIMAHORA | Pedro Sánchez cierra parte del espacio marítimo de Lanzarote hasta el 31 de agosto para su recreo vacacional en La Mareta. El Gobierno del pueblo.",
    "🔴 La extrema derecha esquivando y toreando las difamaciones, insultos y mentiras de los zurdos, óleo sobre lienzo",
    "Ocupan el piso de una mujer de 90 años, heredado de su hijo fallecido, y al acomplejado de Agente Anacleto no se le ocurre otra cosa que justificarlo. Es repugnante tener que aguantar a estos mercenarios sin decencia alguna, haciendo el papel de siervos del Gobierno."
]

if __name__ == '__main__':
    print(f"Running analysis on {len(posts)} embedded default posts (fast mode)")
    
    # Initialize enhanced analyzer in fast mode (no LLM, no retrieval)
    analyzer = EnhancedAnalyzer(use_llm=False, journalism_mode=True)
    
    for i, post in enumerate(posts, 1):
        print(f"\n--- Analyzing Post {i}/{len(posts)} ---")
        result = analyzer.analyze_post(post, retrieve_evidence=False)
        print(f"Category: {result.primary_topic}")
        print(f"Far-right score: {result.far_right_score:.3f}")
        print(f"Threat level: {result.threat_level}")
        print(f"Claims: {result.total_claims}")
