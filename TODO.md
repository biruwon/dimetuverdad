# dimetuverdad - TODO List

## 🚀 High Priority

Top priority:

- disinformation to display links or how it was verified
- [M] Improve LLM explanation
- Local multi modal processing videos
- Posts with links, to use retrieval to fetch their info
- [M] Disinformation/political retrieval to get up to date data/checks
- What categories do we need?
- [L] **Additional platforms**: Extend beyond Twitter/X to Telegram, Facebook, Instagram. We need to use post_id instead of tweet_id
- [L] **Newspaper integration**: Monitor news sources
- [M] **Historical data import**: Import older tweets for trend analysis
- [L] Edited recent posts, how will affect fetching? And show the original
- [L] Allow people to submit account to analyze or post to analyze
- [M] Automate fetcher between new accounts vs latest
- [L] Blog explanining how disinformation works easier to understand
- [L] Instagram videos with avatar explaning disinformation
- [L] Detect topic of the day/week: which accounts promote it, real data, what do they target
- [M] **Pattern refinement**: Review and improve pattern matching rules based on analysis results
- [L] **A/B testing framework**: Compare different LLM models and prompting strategies
- Post deletion: Mostrar una sección de post eliminados, darle más visibilidad, que muestre el media o "hacer screenshoot", ex:  1962875705101869390
- Improve styles and split css/js/html
- Get ready to deploy to production and deploy it

Fixes:
- Post with sensitive data, not displayed because we can't verify age on web. Ex: 1976237480170213419
- Post 1983640141207105603 it's a response but it seems we get content from original post

Random ideas:

- Avatar que discute contigo sobre política sin sesgos

### Model & Analysis Quality
- [M] **False positive reduction**: Improve classification accuracy, especially for political_general vs extremist categories
- [M] **Spanish language optimization**: Fine-tune models specifically for Spanish political discourse
- [L] **Context awareness**: Implement thread/conversation context for better analysis

## 🔧 Medium Priority

### Analysis Engine Improvements
- [M] **Optimize LLM response times**: Current analysis takes 30-60 seconds per tweet with LLM. Investigate model quantization or alternative models
- [M] **Implement batch processing**: Process multiple tweets simultaneously to improve throughput

### Category System Improvements
- [M] **Category validation**: Validate new categories (nationalism, anti_government, historical_revisionism, political_general) with larger datasets
- [L] **Multi-category support**: Allow tweets to belong to multiple categories
- [L] **Category hierarchy**: Implement parent-child relationships (e.g., hate_speech > xenophobia)
- [L] **Custom category creation**: Allow dynamic addition of new analysis categories

## 🛠️ Low Priority

### Documentation & Maintenance
- [x] **API documentation**: Create comprehensive API docs for all endpoints (skipped)
- [M] **Code coverage**: Increase test coverage to >90%

### Analytics & Reporting
- [L] **Trend analysis**: Implement temporal analysis to track narrative evolution
- [L] **Network analysis**: Analyze account interaction patterns and influence networks
- [L] **Comparative analysis**: Compare different political accounts and their content patterns

### Security & Privacy
- [M] **Access control**: Add authentication and authorization to web interface
- [x] **Rate limiting**: Implement API rate limiting to prevent abuse

## 🐛 Known Issues

### Technical Debt
- [M] **Memory optimization**: Reduce memory usage during large batch analysis

### Analysis Accuracy
- [L] **Sarcasm detection**: Improve detection of ironic/sarcastic content
- [L] **Context dependency**: Handle content that requires external context
- [M] **Language mixing**: Better handling of Catalan, Galician, and other co-official languages
- [M] **Political bias calibration**: Ensure balanced classification across political spectrum

## 📊 Analytics & Metrics to Track

### Content Metrics
- [M] **Category distribution trends**: Track changes in content categories over time
- [M] **Account activity patterns**: Monitor posting frequency and timing
- [M] **Engagement correlation**: Analyze relationship between content type and engagement
- [L] **Platform comparison**: Compare content patterns across different social media platforms

## 🎯 Future Features

### Advanced Analysis
- [L] **Sentiment analysis**: Add emotional tone detection beyond category classification
- [L] **Topic modeling**: Implement unsupervised topic discovery
- [L] **Event detection**: Automatically identify significant political events
- [L] **Predictive modeling**: Forecast content trends and narrative shifts

### Integration & Automation
- [M] **Slack/Discord alerts**: Real-time notifications for high-priority content
- [M] **IFTTT/Zapier integration**: Connect with external workflow automation tools
- [L] **Research paper generation**: Automated analysis summary reports
- [L] **Academic partnership**: Integration with university research databases

- Multi language support