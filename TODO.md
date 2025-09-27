# DiMeTuVerdad - TODO List

## 🚀 High Priority

Right now:

- Edited recent posts, how will affect fetching? And show the original
- Improve how to see RTs
- Disinformation and political posts retrieval data to be up to date
- Analyze images and videos, not just text
- Allow people to submit account to analyze or post to analyze
- Improve LLM explanation
- Fetch new tweets only

✅ **COMPLETED**: Admin access to allow reanalyze posts - Simple but secure admin system implemented with token-based authentication, reanalysis tools, and content editing capabilities.

- [ ] **Pattern refinement**: Review and improve pattern matching rules based on analysis results
- [ ] **A/B testing framework**: Compare different LLM models and prompting strategies

### Model & Analysis Quality
- [ ] **False positive reduction**: Improve classification accuracy, especially for political_general vs extremist categories
- [ ] **Spanish language optimization**: Fine-tune models specifically for Spanish political discourse
- [ ] **Context awareness**: Implement thread/conversation context for better analysis

### Data Collection & Sources
- [ ] **Additional platforms**: Extend beyond Twitter/X to Telegram, Facebook, Instagram
- [ ] **Newspaper integration**: Monitor news sources
- [ ] **Historical data import**: Import older tweets for trend analysis


## 🔧 Medium Priority

### Analysis Engine Improvements
- [ ] **Optimize LLM response times**: Current analysis takes 30-60 seconds per tweet with LLM. Investigate model quantization or alternative models
- [ ] **Implement batch processing**: Process multiple tweets simultaneously to improve throughput

### Database & Performance
- [ ] **Database indexing**: Add proper indexes on frequently queried columns (username, category, analysis_timestamp)
- [ ] **Analysis result caching**: Cache LLM results to avoid re-analysis of identical content
- [ ] **Database cleanup**: Remove duplicate tweets and optimize storage
- [ ] **Incremental analysis**: Only analyze new tweets instead of full database scans

### Web Interface Enhancements
- [ ] **Advanced filtering**: Add date range, confidence level, and analysis method filters
- [ ] **Export functionality**: Allow CSV/JSON export of analysis results
- [ ] **User feedback system**: Allow users to flag incorrect classifications for model improvement

### Category System Improvements
- [ ] **Category validation**: Validate new categories (nationalism, anti_government, historical_revisionism, political_general) with larger datasets
- [ ] **Multi-category support**: Allow tweets to belong to multiple categories
- [ ] **Category hierarchy**: Implement parent-child relationships (e.g., hate_speech > xenophobia)
- [ ] **Custom category creation**: Allow dynamic addition of new analysis categories

## 🛠️ Low Priority

### Documentation & Maintenance
- [ ] **API documentation**: Create comprehensive API docs for all endpoints
- [ ] **Code coverage**: Increase test coverage to >90%
- [ ] **Performance benchmarks**: Establish baseline performance metrics
- [ ] **Deployment guide**: Docker containerization and deployment instructions

### Analytics & Reporting
- [ ] **Trend analysis**: Implement temporal analysis to track narrative evolution
- [ ] **Network analysis**: Analyze account interaction patterns and influence networks
- [ ] **Comparative analysis**: Compare different political accounts and their content patterns

### Security & Privacy
- [ ] **Access control**: Add authentication and authorization to web interface
- [ ] **Rate limiting**: Implement API rate limiting to prevent abuse

## 🐛 Known Issues

### Technical Debt
- [ ] **Error handling**: Improve error handling and user feedback in web interface
- [ ] **Memory optimization**: Reduce memory usage during large batch analysis
- [ ] **Code duplication**: Refactor duplicate code between analysis components

### Data Quality Issues
- [ ] **Tweet text truncation**: Handle tweets longer than database field limits
- [ ] **Emoji and special characters**: Improve handling of Unicode characters in analysis
- [ ] **URL expansion**: Resolve shortened URLs to analyze linked content

### Analysis Accuracy
- [ ] **Sarcasm detection**: Improve detection of ironic/sarcastic content
- [ ] **Context dependency**: Handle content that requires external context
- [ ] **Language mixing**: Better handling of Catalan, Galician, and other co-official languages
- [ ] **Political bias calibration**: Ensure balanced classification across political spectrum

## 📊 Analytics & Metrics to Track

### Performance Metrics
- [ ] **Analysis throughput**: Tweets processed per hour/minute
- [ ] **Classification accuracy**: Manual validation of random samples
- [ ] **Resource usage**: CPU, memory, and storage consumption

### Content Metrics
- [ ] **Category distribution trends**: Track changes in content categories over time
- [ ] **Account activity patterns**: Monitor posting frequency and timing
- [ ] **Engagement correlation**: Analyze relationship between content type and engagement
- [ ] **Platform comparison**: Compare content patterns across different social media platforms

## 🎯 Future Features

### Advanced Analysis
- [ ] **Sentiment analysis**: Add emotional tone detection beyond category classification
- [ ] **Topic modeling**: Implement unsupervised topic discovery
- [ ] **Event detection**: Automatically identify significant political events
- [ ] **Predictive modeling**: Forecast content trends and narrative shifts

### Integration & Automation
- [ ] **Slack/Discord alerts**: Real-time notifications for high-priority content
- [ ] **IFTTT/Zapier integration**: Connect with external workflow automation tools
- [ ] **Research paper generation**: Automated analysis summary reports
- [ ] **Academic partnership**: Integration with university research databases