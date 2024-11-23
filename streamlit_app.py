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
from fpdf import FPDF
import langdetect
from datetime import datetime

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
    "risk_assessment": st.sidebar.checkbox("Risk Assessment"),
    "severity_detection": st.sidebar.checkbox("Severity Detection"),
    "critical_keywords": st.sidebar.checkbox("Critical Keyword Identification"),
    "escalation_trigger": st.sidebar.checkbox("Escalation Trigger Detection"),
    "culprit_identification": st.sidebar.checkbox("Culprit Identification"),
    "email_summary": st.sidebar.checkbox("Email Summary"),
    "language_detection": st.sidebar.checkbox("Language Detection"),
    "entity_recognition": st.sidebar.checkbox("Entity Recognition"),
    "response_time_analysis": st.sidebar.checkbox("Response Time Analysis"),
    "attachment_analysis": st.sidebar.checkbox("Attachment Analysis"),
    "customer_tone_analysis": st.sidebar.checkbox("Customer Tone Analysis"),
    "department_identification": st.sidebar.checkbox("Department Identification"),
    "priority_identification": st.sidebar.checkbox("Priority Identification"),
    "urgency_assessment": st.sidebar.checkbox("Urgency Assessment"),
    "action_item_priority": st.sidebar.checkbox("Action Item Priority"),
    "deadline_detection": st.sidebar.checkbox("Deadline Detection"),
    "email_chain_analysis": st.sidebar.checkbox("Email Chain Analysis"),
    "executive_summary": st.sidebar.checkbox("Executive Summary"),
    "actionable_resolution": st.sidebar.checkbox("Actionable Resolution Detection"),
    "response_completeness": st.sidebar.checkbox("Response Completeness"),
    "agreement_identification": st.sidebar.checkbox("Agreement Identification"),
    "feedback_analysis": st.sidebar.checkbox("Feedback Analysis"),
    "threat_detection": st.sidebar.checkbox("Threat Detection"),
    "response_quality_assessment": st.sidebar.checkbox("Response Quality Assessment"),
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
    if "lack of communication" in text.lower():
        return "Root Cause: Lack of communication between teams."
    elif "delayed response" in text.lower():
        return "Root Cause: Delayed response from the team."
    return "Root Cause: Unknown"

# Risk Assessment
def assess_risk(text):
    if "urgent" in text.lower():
        return "Risk Assessment: High risk due to urgency of the issue."
    return "Risk Assessment: Normal risk."

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

# Escalation Trigger Detection
def detect_escalation_trigger(text):
    if "escalate" in text.lower() or "critical" in text.lower():
        return "Escalation Trigger: Immediate escalation required."
    return "Escalation Trigger: No immediate escalation needed."

# Culprit Identification
def identify_culprit(text):
    if "manager" in text.lower():
        return "Culprit: The manager might be responsible."
    elif "team" in text.lower():
        return "Culprit: The team might be responsible."
    return "Culprit: Unknown"

# Language Detection
def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except Exception as e:
        return "Unknown"

# Entity Recognition (dummy for illustration)
def entity_recognition(text):
    entities = ["Email", "Action", "Team", "Manager"]
    return entities

# Response Time Analysis
def response_time_analysis(text):
    if "responded" in text.lower():
        return "Response Time Analysis: Response time is within acceptable range."
    return "Response Time Analysis: Response time is not clear."

# Attachment Analysis (Dummy)
def attachment_analysis(text):
    if "attached" in text.lower():
        return "Attachment Analysis: The email contains an attachment."
    return "Attachment Analysis: No attachment found."

# Customer Tone Analysis
def customer_tone_analysis(text):
    positive_tone_keywords = ["thank you", "appreciate", "grateful"]
    negative_tone_keywords = ["disappointed", "frustrated", "unhappy"]
    tone = "Neutral"
    if any(word in text.lower() for word in positive_tone_keywords):
        tone = "Positive"
    elif any(word in text.lower() for word in negative_tone_keywords):
        tone = "Negative"
    return f"Customer Tone Analysis: {tone}"

# Department Identification (Dummy)
def department_identification(text):
    if "sales" in text.lower():
        return "Department Identification: Sales"
    elif "support" in text.lower():
        return "Department Identification: Support"
    return "Department Identification: Unknown"

# Priority Identification
def identify_priority(text):
    if "high priority" in text.lower():
        return "Priority: High"
    elif "low priority" in text.lower():
        return "Priority: Low"
    return "Priority: Normal"

# Urgency Assessment
def assess_urgency(text):
    if "urgent" in text.lower():
        return "Urgency Assessment: High urgency."
    return "Urgency Assessment: Normal urgency."

# Action Item Priority
def action_item_priority(text):
    if "urgent" in text.lower() or "immediate" in text.lower():
        return "Action Item Priority: High"
    return "Action Item Priority: Normal"

# Deadline Detection
def detect_deadline(text):
    deadlines = ["due", "deadline", "by"]
    if any(word in text.lower() for word in deadlines):
        return "Deadline Detection: Contains a deadline."
    return "Deadline Detection: No deadline mentioned."

# Email Chain Analysis
def email_chain_analysis(text):
    if "forwarded" in text.lower() or "re:" in text.lower():
        return "Email Chain Analysis: This is part of an email chain."
    return "Email Chain Analysis: This email is standalone."

# Executive Summary
def executive_summary(text):
    return f"Executive Summary: {text[:200]}..."

# Actionable Resolution Detection
def actionable_resolution(text):
    if "resolve" in text.lower() or "solution" in text.lower():
        return "Actionable Resolution: The email includes a resolution or solution."
    return "Actionable Resolution: No actionable resolution found."

# Response Completeness
def response_completeness(text):
    if "thank you" in text.lower() and "best regards" in text.lower():
        return "Response Completeness: Response is complete."
    return "Response Completeness: Response is incomplete."

# Agreement Identification
def agreement_identification(text):
    if "agree" in text.lower():
        return "Agreement Identification: The email includes an agreement."
    return "Agreement Identification: No agreement found."

# Feedback Analysis
def feedback_analysis(text):
    if "feedback" in text.lower():
        return "Feedback Analysis: Feedback present."
    return "Feedback Analysis: No feedback found."

# Threat Detection
def threat_detection(text):
    threat_keywords = ["threat", "warning", "danger"]
    if any(word in text.lower() for word in threat_keywords):
        return "Threat Detection: Threat detected."
    return "Threat Detection: No threat detected."

# Response Quality Assessment
def response_quality_assessment(text):
    if len(text.split()) < 20:
        return "Response Quality: Incomplete response."
    return "Response Quality: Complete response."

# Visualizing Word Frequency
def visualize_word_frequency(word_counts):
    plt.figure(figsize=(10, 5))
    plt.bar(word_counts.keys(), word_counts.values())
    plt.xticks(rotation=45)
    plt.title("Word Frequency")
    plt.tight_layout()

# Exporting Insights as JSON, Text, and PDF
def export_insights(text_data, summary):
    pdf_buffer = BytesIO(export_pdf(text_data))
    buffer_txt = BytesIO(text_data.encode("utf-8"))
    buffer_json = BytesIO(json.dumps(summary, indent=4).encode("utf-8"))
    st.download_button("Download as Text", data=buffer_txt, file_name="analysis.txt", mime="text/plain")
    st.download_button("Download as PDF", data=pdf_buffer, file_name="analysis.pdf", mime="application/pdf")
    st.download_button("Download as JSON", data=buffer_json, file_name="analysis.json", mime="application/json")

# File upload for email content
uploaded_file = st.file_uploader("Upload Email File", type=["txt", "eml"])

if uploaded_file is not None:
    email_content = uploaded_file.read().decode("utf-8")

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

        if features["escalation_trigger"]:
            escalation_trigger = detect_escalation_trigger(email_content)
            st.subheader("Escalation Trigger")
            st.write(escalation_trigger)

        if features["culprit_identification"]:
            culprit = identify_culprit(email_content)
            st.subheader("Culprit Identification")
            st.write(culprit)

        if features["response_time_analysis"]:
            response_time = response_time_analysis(email_content)
            st.subheader("Response Time Analysis")
            st.write(response_time)

        if features["attachment_analysis"]:
            attachment = attachment_analysis(email_content)
            st.subheader("Attachment Analysis")
            st.write(attachment)

        if features["customer_tone_analysis"]:
            tone = customer_tone_analysis(email_content)
            st.subheader("Customer Tone Analysis")
            st.write(tone)

        if features["department_identification"]:
            department = department_identification(email_content)
            st.subheader("Department Identification")
            st.write(department)

        if features["priority_identification"]:
            priority = identify_priority(email_content)
            st.subheader("Priority Identification")
            st.write(priority)

        if features["urgency_assessment"]:
            urgency = assess_urgency(email_content)
            st.subheader("Urgency Assessment")
            st.write(urgency)

        if features["action_item_priority"]:
            item_priority = action_item_priority(email_content)
            st.subheader("Action Item Priority")
            st.write(item_priority)

        if features["deadline_detection"]:
            deadline = detect_deadline(email_content)
            st.subheader("Deadline Detection")
            st.write(deadline)

        if features["email_chain_analysis"]:
            email_chain = email_chain_analysis(email_content)
            st.subheader("Email Chain Analysis")
            st.write(email_chain)

        if features["executive_summary"]:
            executive = executive_summary(email_content)
            st.subheader("Executive Summary")
            st.write(executive)

        if features["actionable_resolution"]:
            resolution = actionable_resolution(email_content)
            st.subheader("Actionable Resolution Detection")
            st.write(resolution)

        if features["response_completeness"]:
            completeness = response_completeness(email_content)
            st.subheader("Response Completeness")
            st.write(completeness)

        if features["agreement_identification"]:
            agreement = agreement_identification(email_content)
            st.subheader("Agreement Identification")
            st.write(agreement)

        if features["feedback_analysis"]:
            feedback = feedback_analysis(email_content)
            st.subheader("Feedback Analysis")
            st.write(feedback)

        if features["threat_detection"]:
            threat = threat_detection(email_content)
            st.subheader("Threat Detection")
            st.write(threat)

        if features["response_quality_assessment"]:
            quality = response_quality_assessment(email_content)
            st.subheader("Response Quality Assessment")
            st.write(quality)

        # Export options
        export_content = f"Summary:\n{summary}\n\nResponse:\n{response}\n\nHighlights:\n{highlights}\n\nSentiment Analysis: {sentiment_label} (Score: {sentiment})"
        export_insights(export_content, {"summary": summary, "response": response, "highlights": highlights, "sentiment": sentiment_label})

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Paste email content or upload an email file and click 'Generate Insights' to start.")
