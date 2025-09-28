# dimetuverdad - TODO List

## 🚀 High Priority

Top priority:

- [L] Edited recent posts, how will affect fetching? And show the original
- [M] Improve how to see RTs
- [M] Disinformation and political posts retrieval data to be up to date
- [L] Analyze images and videos, not just text
- [L] Allow people to submit account to analyze or post to analyze
- [M] Improve LLM explanation
- [M] Fetch new tweets only
- [L] Blog/instagram explanining how disinformation works easier to understand
- [L] Detect topic of the day/week: which accounts promote it, real data, what do they target

- [M] **Pattern refinement**: Review and improve pattern matching rules based on analysis results
- [L] **A/B testing framework**: Compare different LLM models and prompting strategies

### Model & Analysis Quality
- [M] **False positive reduction**: Improve classification accuracy, especially for political_general vs extremist categories
- [M] **Spanish language optimization**: Fine-tune models specifically for Spanish political discourse
- [L] **Context awareness**: Implement thread/conversation context for better analysis

### Data Collection & Sources
- [L] **Additional platforms**: Extend beyond Twitter/X to Telegram, Facebook, Instagram
- [L] **Newspaper integration**: Monitor news sources
- [M] **Historical data import**: Import older tweets for trend analysis


## 🔧 Medium Priority

### Analysis Engine Improvements
- [M] **Optimize LLM response times**: Current analysis takes 30-60 seconds per tweet with LLM. Investigate model quantization or alternative models
- [M] **Implement batch processing**: Process multiple tweets simultaneously to improve throughput

### Database & Performance
- [M] **Analysis result caching**: Cache LLM results to avoid re-analysis of identical content
- [M] **Incremental analysis**: Only analyze new tweets instead of full database scans

### Web Interface Enhancements
- [M] **Advanced filtering**: Add date range, confidence level, and analysis method filters
- [M] **User feedback system**: Allow users to flag incorrect classifications for model improvement

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
- [S] **Rate limiting**: Implement API rate limiting to prevent abuse

## 🐛 Known Issues

### Technical Debt
- [S] **Error handling**: Improve error handling and user feedback in web interface
- [M] **Memory optimization**: Reduce memory usage during large batch analysis
- [S] **Code duplication**: Refactor duplicate code between analysis components

### Data Quality Issues
- [S] **Tweet text truncation**: Handle tweets longer than database field limits
- [S] **Emoji and special characters**: Improve handling of Unicode characters in analysis
- [S] **URL expansion**: Resolve shortened URLs to analyze linked content

### Analysis Accuracy
- [L] **Sarcasm detection**: Improve detection of ironic/sarcastic content
- [L] **Context dependency**: Handle content that requires external context
- [M] **Language mixing**: Better handling of Catalan, Galician, and other co-official languages
- [M] **Political bias calibration**: Ensure balanced classification across political spectrum

## 📊 Analytics & Metrics to Track

### Performance Metrics
- [S] **Analysis throughput**: Tweets processed per hour/minute
- [S] **Classification accuracy**: Manual validation of random samples
- [S] **Resource usage**: CPU, memory, and storage consumption

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