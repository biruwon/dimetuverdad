"""Run analysis on embedded default posts (no DB) in fast mode.
This script imports the posts array from analyze_posts.py and calls main with skip_retrieval=True and skip_save=True
"""
from analyze_posts import main

# Default/example posts kept separate from the analysis module
posts = [
    "Prohíben limpiar y desbrozar el monte, y si lo haces te multan. Y ahora te van a investigar por no haberlo limpiado.",
    "Hoy hace sol en Madrid.",
    "🔴 #ÚLTIMAHORA | Pedro Sánchez cierra parte del espacio marítimo de Lanzarote hasta el 31 de agosto para su recreo vacacional en La Mareta. El Gobierno del pueblo.",
    "🔴 La “extrema derecha” esquivando y toreando las difamaciones, insultos y mentiras de los zurdos, óleo sobre lienzo",
    "Ocupan el piso de una mujer de 90 años, heredado de su hijo fallecido, y al acomplejado de Agente Anacleto no se le ocurre otra cosa que justificarlo. Es repugnante tener que aguantar a estos mercenarios sin decencia alguna, haciendo el papel de siervos del Gobierno."
]

if __name__ == '__main__':
    print(f"Running analysis on {len(posts)} embedded default posts (fast mode)")
    main(posts, skip_retrieval=True, skip_save=True)
