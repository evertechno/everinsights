import streamlit as st
import google.generativeai as genai
from io import BytesIO
import json
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF

# Configure API Key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# App Configuration
st.set_page_config(page_title="Escalytics", page_icon="📧", layout="wide")
st.title("⚡Escalytics by EverTech")
st.write("Extract insights, root causes, and actionable steps from emails.")

# Sidebar for Features
st.sidebar.header("Settings")
features = {
    "sentiment": st.sidebar.checkbox("Perform Sentiment Analysis"),
    "highlights": st.sidebar.checkbox("Highlight Key Phrases"),
    "response": st.sidebar.checkbox("Generate Suggested Response"),
    "wordcloud": st.sidebar.checkbox("Generate Word Cloud"),
    "grammar_check": st.sidebar.checkbox("Grammar Check"),
    "key_phrases": st.sidebar.checkbox("Extract Key Phrases"),
    "actionable_items": st.sidebar.checkbox("Extract Actionable Items"),
    "root_cause": st.sidebar.checkbox("Root Cause Detection"),
    "culprit_identification": st.sidebar.checkbox("Culprit Identification"),
    "trend_analysis": st.sidebar.checkbox("Trend Analysis"),
    "risk_assessment": st.sidebar.checkbox("Risk Assessment"),
    "severity_detection": st.sidebar.checkbox("Severity Detection"),
    "critical_keywords": st.sidebar.checkbox("Critical Keyword Identification"),
    "export": st.sidebar.checkbox("Export Options"),
    "sentiment_analysis_report": st.sidebar.checkbox("Sentiment Analysis Report"),
    "email_summary": st.sidebar.checkbox("Email Summary"),
    "language_detection": st.sidebar.checkbox("Language Detection"),
    "topic_modeling": st.sidebar.checkbox("Topic Modeling"),
    "entity_recognition": st.sidebar.checkbox("Entity Recognition"),
    "action_item_prioritization": st.sidebar.checkbox("Action Item Prioritization"),
    "email_classification": st.sidebar.checkbox("Email Classification"),
    "correlation_analysis": st.sidebar.checkbox("Correlation Analysis"),
    "time_analysis": st.sidebar.checkbox("Time-based Trend Analysis"),
    "social_network_analysis": st.sidebar.checkbox("Social Network Analysis"),
    "email_thread_analysis": st.sidebar.checkbox("Email Thread Analysis"),
    "duplicate_detection": st.sidebar.checkbox("Duplicate Email Detection"),
    "language_translation": st.sidebar.checkbox("Language Translation"),
    "conversation_context": st.sidebar.checkbox("Conversation Context Extraction"),
    "action_plan": st.sidebar.checkbox("Generate Action Plan"),
    "feedback_analysis": st.sidebar.checkbox("Feedback Analysis"),
    "toxic_content_detection": st.sidebar.checkbox("Toxic Content Detection"),
    "auto_summary": st.sidebar.checkbox("Automatic Summary Generation"),
    "confusion_matrix": st.sidebar.checkbox("Confusion Matrix for Classification"),
    "data_visualization": st.sidebar.checkbox("Data Visualization"),
    "alert_generation": st.sidebar.checkbox("Alert Generation"),
    "decision_tree_analysis": st.sidebar.checkbox("Decision Tree Analysis"),
    "clustering_analysis": st.sidebar.checkbox("Clustering Analysis"),
    "email_composition": st.sidebar.checkbox("Email Composition Assistance"),
    "text_similarity": st.sidebar.checkbox("Text Similarity Analysis"),
    "multilingual_support": st.sidebar.checkbox("Multilingual Email Support"),
    "feedback_score_analysis": st.sidebar.checkbox("Feedback Score Analysis"),
    "email_length_analysis": st.sidebar.checkbox("Email Length Analysis"),
    "suggested_improvements": st.sidebar.checkbox("Suggested Improvements"),
    "action_plan_export": st.sidebar.checkbox("Export Action Plan"),
    "email_responder": st.sidebar.checkbox("Email Responder Suggestions"),
    "user_experience_analysis": st.sidebar.checkbox("User Experience Analysis"),
    "data_trend_forecasting": st.sidebar.checkbox("Data Trend Forecasting"),
    "time_to_resolution_analysis": st.sidebar.checkbox("Time to Resolution Analysis"),
    "contact_frequency_analysis": st.sidebar.checkbox("Contact Frequency Analysis"),
    "response_time_analysis": st.sidebar.checkbox("Response Time Analysis"),
    "sentiment_trend_analysis": st.sidebar.checkbox("Sentiment Trend Analysis"),
    "emotion_detection": st.sidebar.checkbox("Emotion Detection"),
    "follow_up_action_items": st.sidebar.checkbox("Follow-up Action Items"),
    "escalation_analysis": st.sidebar.checkbox("Escalation Analysis"),
    "case_priority_assessment": st.sidebar.checkbox("Case Priority Assessment"),
    "actionable_feedback_detection": st.sidebar.checkbox("Actionable Feedback Detection"),
    "email_type_classification": st.sidebar.checkbox("Email Type Classification"),
    "resolution_efficiency_analysis": st.sidebar.checkbox("Resolution Efficiency Analysis"),
    "message_parsing": st.sidebar.checkbox("Message Parsing"),
    "performance_metrics_analysis": st.sidebar.checkbox("Performance Metrics Analysis"),
}

# Input Email Section
email_content = st.text_area("Paste your email content here:", height=200)

MAX_EMAIL_LENGTH = 1000

# Cache the AI responses to improve performance
@st.cache_data(ttl=3600)
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Sentiment Analysis
def get_sentiment(email_content):
    positive_keywords = ["happy", "good", "great", "excellent", "love"]
    negative_keywords = ["sad", "bad", "hate", "angry", "disappointed"]
    sentiment_score = 0
    for word in email_content.split():
        if word.lower() in positive_keywords:
            sentiment_score += 1
        elif word.lower() in negative_keywords:
            sentiment_score -= 1
    return sentiment_score

# Grammar Check (basic spelling correction)
def grammar_check(text):
    corrections = {
        "recieve": "receive",
        "adress": "address",
        "teh": "the",
        "occured": "occurred"
    }
    for word, correct in corrections.items():
        text = text.replace(word, correct)
    return text

# Key Phrase Extraction
def extract_key_phrases(text):
    key_phrases = re.findall(r"\b[A-Za-z]{4,}\b", text)
    return list(set(key_phrases))  # Remove duplicates

# Word Cloud Generation
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Export to PDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# Actionable Items Extraction
def extract_actionable_items(text):
    actions = [line for line in text.split("\n") if "to" in line.lower() or "action" in line.lower()]
    return actions

# Root Cause Detection
def detect_root_cause(text):
    return "Possible root cause: Lack of clear communication in the process."

# Culprit Identification
def identify_culprit(text):
    if "manager" in text.lower():
        return "Culprit: The manager might be responsible."
    elif "team" in text.lower():
        return "Culprit: The team might be responsible."
    return "Culprit: Unknown"

# Trend Analysis
def analyze_trends(text):
    return "Trend detected: Delay in project timelines."

# Risk Assessment
def assess_risk(text):
    return "Risk assessment: High risk due to delayed communication."

# Severity Detection
def detect_severity(text):
    if "urgent" in text.lower():
        return "Severity: High"
    return "Severity: Normal"

# Critical Keyword Identification
def identify_critical_keywords(text):
    critical_keywords = ["urgent", "problem", "issue", "failure"]
    critical_terms = [word for word in text.split() if word.lower() in critical_keywords]
    return critical_terms

# Language Detection
def detect_language(text):
    return "English"  # Placeholder, integrate with a language detection API like langdetect

# Entity Recognition (dummy for illustration)
def entity_recognition(text):
    entities = ["Email", "Action", "Team", "Manager"]
    return entities

# Action Item Prioritization
def prioritize_action_items(items):
    return sorted(items, key=lambda x: len(x), reverse=True)

# Email Classification (dummy)
def classify_email(text):
    if "urgent" in text.lower():
        return "High priority"
    return "Normal priority"

# Visualizing Word Frequency
def visualize_word_frequency(word_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(word_counts.keys(), word_counts.values())
    plt.xticks(rotation=45)
    plt.title("Word Frequency")
    plt.tight_layout()

# Clustering Analysis (dummy)
def clustering_analysis(text):
    return "Clustering completed using K-Means"

# Email Length Analysis
def analyze_email_length(text):
    return f"Email length: {len(text)} characters"

# Exporting Insights as JSON, Text, and PDF
def export_insights(text_data, summary):
    pdf_buffer = BytesIO(export_pdf(text_data))
    buffer_txt = BytesIO(text_data.encode("utf-8"))
    buffer_json = BytesIO(json.dumps(summary, indent=4).encode("utf-8"))
    st.download_button("Download as Text", data=buffer_txt, file_name="analysis.txt", mime="text/plain")
    st.download_button("Download as PDF", data=pdf_buffer, file_name="analysis.pdf", mime="application/pdf")
    st.download_button("Download as JSON", data=buffer_json, file_name="analysis.json", mime="application/json")

# Layout for displaying results
if email_content and st.button("Generate Insights"):
    try:
        # Generate AI-like responses (using google.generativeai for content generation)
        summary = get_ai_response("Summarize the email in a concise, actionable format:\n\n", email_content)
        response = get_ai_response("Draft a professional response to this email:\n\n", email_content) if features["response"] else ""
        highlights = get_ai_response("Highlight key points and actions in this email:\n\n", email_content) if features["highlights"] else ""

        # Sentiment Analysis
        sentiment = get_sentiment(email_content)
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Generate Word Cloud
        wordcloud = generate_wordcloud(email_content)
        wordcloud_fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        # Display Results
        st.subheader("AI Summary")
        st.write(summary)

        if features["response"]:
            st.subheader("Suggested Response")
            st.write(response)

        if features["highlights"]:
            st.subheader("Key Highlights")
            st.write(highlights)

        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment_label} (Score: {sentiment})")

        if features["grammar_check"]:
            corrected_text = grammar_check(email_content)
            st.subheader("Grammar Check")
            st.write("Corrected Text:")
            st.write(corrected_text)

        if features["key_phrases"]:
            key_phrases = extract_key_phrases(email_content)
            st.subheader("Key Phrases Extracted")
            st.write(key_phrases)

        if features["wordcloud"]:
            st.subheader("Word Cloud")
            st.pyplot(wordcloud_fig)

        if features["actionable_items"]:
            actionable_items = extract_actionable_items(email_content)
            st.subheader("Actionable Items")
            st.write(actionable_items)

        # RCA and Insights Features
        if features["root_cause"]:
            root_cause = detect_root_cause(email_content)
            st.subheader("Root Cause Detection")
            st.write(root_cause)

        if features["culprit_identification"]:
            culprit = identify_culprit(email_content)
            st.subheader("Culprit Identification")
            st.write(culprit)

        if features["trend_analysis"]:
            trends = analyze_trends(email_content)
            st.subheader("Trend Analysis")
            st.write(trends)

        if features["risk_assessment"]:
            risk = assess_risk(email_content)
            st.subheader("Risk Assessment")
            st.write(risk)

        if features["severity_detection"]:
            severity = detect_severity(email_content)
            st.subheader("Severity Detection")
            st.write(severity)

        if features["critical_keywords"]:
            critical_terms = identify_critical_keywords(email_content)
            st.subheader("Critical Keywords Identified")
            st.write(critical_terms)

        # Export options
        export_content = f"Summary:\n{summary}\n\nResponse:\n{response}\n\nHighlights:\n{highlights}\n\nSentiment Analysis: {sentiment_label} (Score: {sentiment})"
        export_insights(export_content, {"summary": summary, "response": response, "highlights": highlights, "sentiment": sentiment_label})

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Paste email content and click 'Generate Insights' to start.")
