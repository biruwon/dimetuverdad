"""
Real-time Monitoring Dashboard for Far-Right Analysis
Provides continuous monitoring and alert capabilities
"""

import json
import time
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
from collections import defaultdict, deque
import signal
import sys

from enhanced_analyzer import EnhancedAnalyzer, AnalysisResult

DB_PATH = os.path.join(os.path.dirname(__file__), 'accounts.db')

class RealTimeMonitor:
    """
    Real-time monitoring system for far-right activism analysis.
    Features:
    - Continuous database monitoring
    - Real-time threat detection
    - Alert system
    - Trend analysis
    - Dashboard display
    """
    
    def __init__(self, update_interval: int = 30):
        self.analyzer = EnhancedAnalyzer(use_llm=False)  # Fast analysis for real-time
        self.update_interval = update_interval
        self.running = False
        
        # Monitoring state
        self.recent_analyses = deque(maxlen=1000)  # Keep last 1000 analyses
        self.alert_history = deque(maxlen=100)     # Keep last 100 alerts
        self.threat_trends = defaultdict(list)     # Track threat trends
        self.last_check_time = datetime.now()
        
        # Alert thresholds
        self.alert_config = {
            'critical_threat_threshold': 0.8,
            'high_risk_threshold': 0.6,
            'burst_detection_window': 300,  # 5 minutes
            'burst_threshold': 5,            # 5 posts in window
            'trend_window': 3600,            # 1 hour
            'escalation_threshold': 0.2      # 20% increase
        }
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'threats_detected': 0,
            'alerts_generated': 0,
            'last_update': None,
            'processing_rate': 0.0
        }
        
        print("🔍 SISTEMA DE MONITORIZACIÓN EN TIEMPO REAL")
        print("=" * 50)
        print("Configuración:")
        print(f"   • Intervalo de actualización: {update_interval}s")
        print(f"   • Umbral amenaza crítica: {self.alert_config['critical_threat_threshold']}")
        print(f"   • Ventana detección ráfagas: {self.alert_config['burst_detection_window']}s")
        print(f"   • Umbral escalada: {self.alert_config['escalation_threshold']}")
        print()
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        print("🚀 Iniciando monitorización en tiempo real...")
        print("   Presiona Ctrl+C para detener")
        print("-" * 50)
        
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                start_time = time.time()
                
                # Check for new posts
                new_posts = self._get_new_posts()
                
                if new_posts:
                    # Analyze new posts
                    results = self._analyze_new_posts(new_posts)
                    
                    # Update statistics
                    self._update_statistics(results)
                    
                    # Check for alerts
                    alerts = self._check_alerts(results)
                    
                    # Update trends
                    self._update_trends(results)
                    
                    # Display dashboard
                    self._display_dashboard(results, alerts)
                
                # Sleep until next update
                processing_time = time.time() - start_time
                sleep_time = max(0, self.update_interval - processing_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n📟 Señal recibida ({signum}), cerrando monitorización...")
        self.running = False
    
    def _get_new_posts(self) -> List[Dict]:
        """Get new posts from database since last check."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # Get posts newer than last check
            c.execute('''
            SELECT id, content, author, created_at, url 
            FROM tweets 
            WHERE created_at > ? 
            AND length(content) > 20
            ORDER BY created_at DESC
            ''', (self.last_check_time.isoformat(),))
            
            rows = c.fetchall()
            conn.close()
            
            # Update last check time
            self.last_check_time = datetime.now()
            
            # Convert to list of dicts
            new_posts = []
            for row in rows:
                new_posts.append({
                    'id': row[0],
                    'content': row[1],
                    'author': row[2],
                    'created_at': row[3],
                    'url': row[4] or f"post_{row[0]}"
                })
            
            return new_posts
            
        except Exception as e:
            print(f"❌ Error accediendo a base de datos: {e}")
            return []
    
    def _analyze_new_posts(self, posts: List[Dict]) -> List[AnalysisResult]:
        """Analyze new posts quickly."""
        results = []
        
        for post in posts:
            try:
                # Quick analysis without evidence retrieval
                result = self.analyzer.analyze_post(
                    text=post['content'],
                    retrieve_evidence=False,
                    tweet_url=post['url']
                )
                
                # Add metadata
                result.post_id = post['id']
                result.author = post['author']
                result.created_at = post['created_at']
                
                results.append(result)
                self.recent_analyses.append(result)
                
            except Exception as e:
                print(f"❌ Error analizando post {post['id']}: {e}")
                continue
        
        return results
    
    def _check_alerts(self, results: List[AnalysisResult]) -> List[Dict]:
        """Check for alert conditions."""
        alerts = []
        current_time = datetime.now()
        
        for result in results:
            # Critical threat alert
            if result.far_right_score >= self.alert_config['critical_threat_threshold']:
                alert = {
                    'type': 'AMENAZA_CRITICA',
                    'severity': 'CRITICAL',
                    'timestamp': current_time,
                    'post_id': getattr(result, 'post_id', 'unknown'),
                    'author': getattr(result, 'author', 'unknown'),
                    'score': result.far_right_score,
                    'threat_level': result.threat_level,
                    'message': f'Amenaza crítica detectada: {result.far_right_score:.2f}',
                    'content_preview': result.post_text[:100] + "..." if len(result.post_text) > 100 else result.post_text
                }
                alerts.append(alert)
                self.alert_history.append(alert)
                self.stats['alerts_generated'] += 1
        
        # Burst detection - multiple high-risk posts in short time
        recent_window = current_time - timedelta(seconds=self.alert_config['burst_detection_window'])
        recent_high_risk = [
            r for r in self.recent_analyses 
            if (hasattr(r, 'created_at') and 
                datetime.fromisoformat(r.created_at) > recent_window and
                r.far_right_score > self.alert_config['high_risk_threshold'])
        ]
        
        if len(recent_high_risk) >= self.alert_config['burst_threshold']:
            alert = {
                'type': 'RAFAGA_EXTREMISMO',
                'severity': 'HIGH',
                'timestamp': current_time,
                'post_count': len(recent_high_risk),
                'time_window': self.alert_config['burst_detection_window'],
                'message': f'Ráfaga de {len(recent_high_risk)} posts extremistas en {self.alert_config["burst_detection_window"]}s',
                'avg_score': sum(r.far_right_score for r in recent_high_risk) / len(recent_high_risk)
            }
            alerts.append(alert)
            self.alert_history.append(alert)
            self.stats['alerts_generated'] += 1
        
        return alerts
    
    def _update_trends(self, results: List[AnalysisResult]):
        """Update trend tracking."""
        current_time = datetime.now()
        
        for result in results:
            # Track threat levels over time
            self.threat_trends['threat_scores'].append({
                'timestamp': current_time,
                'score': result.far_right_score,
                'threat_level': result.threat_level
            })
            
            # Track topics
            if result.primary_topic:
                self.threat_trends['topics'].append({
                    'timestamp': current_time,
                    'topic': result.primary_topic
                })
        
        # Clean old trend data (keep last hour)
        cutoff_time = current_time - timedelta(seconds=self.alert_config['trend_window'])
        
        for trend_type in self.threat_trends:
            self.threat_trends[trend_type] = [
                item for item in self.threat_trends[trend_type]
                if item['timestamp'] > cutoff_time
            ]
    
    def _update_statistics(self, results: List[AnalysisResult]):
        """Update monitoring statistics."""
        self.stats['total_processed'] += len(results)
        self.stats['threats_detected'] += len([r for r in results if r.far_right_score > 0.3])
        self.stats['last_update'] = datetime.now()
        
        # Calculate processing rate (posts per minute)
        if len(self.recent_analyses) > 1:
            time_span = (datetime.now() - datetime.fromisoformat(
                getattr(self.recent_analyses[0], 'created_at', datetime.now().isoformat())
            )).total_seconds()
            if time_span > 0:
                self.stats['processing_rate'] = len(self.recent_analyses) / (time_span / 60)
    
    def _display_dashboard(self, new_results: List[AnalysisResult], alerts: List[Dict]):
        """Display real-time dashboard."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🔍 MONITORIZACIÓN EN TIEMPO REAL - EXTREMA DERECHA")
        print("=" * 60)
        print(f"🕒 Última actualización: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Current batch statistics
        if new_results:
            high_risk = [r for r in new_results if r.far_right_score > 0.5]
            threats = [r for r in new_results if r.threat_level in ['HIGH', 'CRITICAL']]
            
            print("📊 ANÁLISIS ACTUAL:")
            print(f"   • Posts nuevos: {len(new_results)}")
            print(f"   • Alto riesgo: {len(high_risk)}")
            print(f"   • Amenazas: {len(threats)}")
            print(f"   • Score máximo: {max((r.far_right_score for r in new_results), default=0.0):.3f}")
        else:
            print("📊 ANÁLISIS ACTUAL: Sin posts nuevos")
        
        print()
        
        # Overall statistics
        print("📈 ESTADÍSTICAS GENERALES:")
        print(f"   • Total procesados: {self.stats['total_processed']}")
        print(f"   • Amenazas detectadas: {self.stats['threats_detected']}")
        print(f"   • Alertas generadas: {self.stats['alerts_generated']}")
        print(f"   • Tasa procesamiento: {self.stats['processing_rate']:.1f} posts/min")
        print()
        
        # Recent alerts
        if self.alert_history:
            print("🚨 ALERTAS RECIENTES:")
            for alert in list(self.alert_history)[-3:]:
                print(f"   • [{alert['timestamp'].strftime('%H:%M:%S')}] {alert['type']}: {alert['message']}")
        else:
            print("🚨 ALERTAS RECIENTES: Ninguna")
        
        print()
        
        # Trends
        if self.threat_trends['threat_scores']:
            recent_scores = [item['score'] for item in self.threat_trends['threat_scores'][-10:]]
            avg_recent = sum(recent_scores) / len(recent_scores)
            
            print("📊 TENDENCIAS (última hora):")
            print(f"   • Score promedio reciente: {avg_recent:.3f}")
            print(f"   • Posts analizados: {len(self.threat_trends['threat_scores'])}")
            
            # Topic distribution
            if self.threat_trends['topics']:
                topic_counts = defaultdict(int)
                for item in self.threat_trends['topics'][-20:]:
                    topic_counts[item['topic']] += 1
                
                print("   • Temas principales:")
                for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"     - {topic}: {count}")
        
        print()
        
        # Recent high-risk posts
        recent_high_risk = [
            r for r in list(self.recent_analyses)[-10:] 
            if r.far_right_score > 0.5
        ]
        
        if recent_high_risk:
            print("⚠️ POSTS DE ALTO RIESGO RECIENTES:")
            for result in recent_high_risk[-3:]:
                author = getattr(result, 'author', 'desconocido')
                preview = result.post_text[:50] + "..." if len(result.post_text) > 50 else result.post_text
                print(f"   • [{author}] {result.far_right_score:.3f} - {preview}")
        else:
            print("⚠️ POSTS DE ALTO RIESGO RECIENTES: Ninguno")
        
        print()
        print(f"🔄 Próxima actualización en {self.update_interval}s...")
        print("   Presiona Ctrl+C para detener")
    
    def _shutdown(self):
        """Graceful shutdown."""
        print("\n🛑 CERRANDO MONITORIZACIÓN")
        print("-" * 30)
        print(f"📊 Estadísticas finales:")
        print(f"   • Total procesados: {self.stats['total_processed']}")
        print(f"   • Amenazas detectadas: {self.stats['threats_detected']}")
        print(f"   • Alertas generadas: {self.stats['alerts_generated']}")
        print(f"   • Tiempo funcionamiento: {datetime.now() - (self.stats['last_update'] or datetime.now())}")
        
        # Save monitoring session report
        try:
            report = {
                'session_end': datetime.now().isoformat(),
                'statistics': self.stats,
                'alert_summary': {
                    'total_alerts': len(self.alert_history),
                    'alert_types': defaultdict(int)
                },
                'trend_summary': {
                    'total_threat_scores': len(self.threat_trends.get('threat_scores', [])),
                    'avg_threat_score': sum(item['score'] for item in self.threat_trends.get('threat_scores', [])) / max(1, len(self.threat_trends.get('threat_scores', []))),
                    'peak_threat_score': max((item['score'] for item in self.threat_trends.get('threat_scores', [])), default=0.0)
                }
            }
            
            # Count alert types
            for alert in self.alert_history:
                report['alert_summary']['alert_types'][alert['type']] += 1
            
            # Save report
            filename = f"monitoring_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 Informe de sesión guardado: {filename}")
            
        except Exception as e:
            print(f"❌ Error guardando informe: {e}")
        
        print("✅ Monitorización finalizada")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Far-Right Monitoring Dashboard')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds')
    parser.add_argument('--critical-threshold', type=float, default=0.8, help='Critical threat threshold')
    parser.add_argument('--burst-threshold', type=int, default=5, help='Burst detection threshold')
    parser.add_argument('--burst-window', type=int, default=300, help='Burst detection window in seconds')
    
    args = parser.parse_args()
    
    # Create and configure monitor
    monitor = RealTimeMonitor(update_interval=args.interval)
    
    # Update configuration
    monitor.alert_config.update({
        'critical_threat_threshold': args.critical_threshold,
        'burst_threshold': args.burst_threshold,
        'burst_detection_window': args.burst_window
    })
    
    # Start monitoring
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
