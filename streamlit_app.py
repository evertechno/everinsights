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
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import hashlib
import os
from datetime import datetime
from typing import List

# Configure API Key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# AES Encryption Key (ensure secure handling of the key in production)
ENCRYPTION_KEY = st.secrets["ENCRYPTION_KEY"]

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
    # New features added:
    "sla_breach": st.sidebar.checkbox("Detect SLA Breach"),
    "prioritization": st.sidebar.checkbox("Prioritize Issues"),
    "resolution_time": st.sidebar.checkbox("Predict Resolution Time"),
    "stakeholder_identification": st.sidebar.checkbox("Identify Stakeholders"),
    "escalation_levels": st.sidebar.checkbox("Define Escalation Levels"),
    "performance_metrics": st.sidebar.checkbox("Extract Performance Metrics"),
    "resolution_recommendations": st.sidebar.checkbox("Suggest Resolutions"),
    "project_impact": st.sidebar.checkbox("Identify Project Impact"),
    "compliance_check": st.sidebar.checkbox("Check Compliance Issues"),
    "incident_tracking": st.sidebar.checkbox("Track Incident Numbers"),
    "customer_satisfaction": st.sidebar.checkbox("Predict Customer Satisfaction"),
    "feedback_loop": st.sidebar.checkbox("Identify Feedback Loops"),
    "automated_response": st.sidebar.checkbox("Generate Automated Responses"),
    "contractual_obligations": st.sidebar.checkbox("Identify Contractual Obligations"),
    "root_cause_frequency": st.sidebar.checkbox("Analyze Root Cause Frequency"),
    "budget_impact": st.sidebar.checkbox("Analyze Budget Impact"),
    "vendor_management": st.sidebar.checkbox("Extract Vendor Information"),
    "supply_chain_issues": st.sidebar.checkbox("Identify Supply Chain Issues"),
    "service_request": st.sidebar.checkbox("Categorize Service Requests"),
    "ticket_escalation": st.sidebar.checkbox("Track Ticket Escalations"),
    "operational_impact": st.sidebar.checkbox("Identify Operational Impact"),
    "regulatory_alerts": st.sidebar.checkbox("Detect Regulatory Alerts"),
    "risk_mitigation": st.sidebar.checkbox("Suggest Risk Mitigation"),
    "business_continuity": st.sidebar.checkbox("Business Continuity Analysis"),
    "market_sentiment": st.sidebar.checkbox("Extract Market Sentiment")
}

# Encryption/Decryption utility functions
def encrypt_data(data: str) -> str:
    """Encrypt data using AES encryption"""
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(ENCRYPTION_KEY), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = data + (16 - len(data) % 16) * " "  # Pad data to be multiple of block size
    encrypted = encryptor.update(padded_data.encode()) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode('utf-8')

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt data using AES decryption"""
    encrypted_data_bytes = base64.b64decode(encrypted_data)
    iv = encrypted_data_bytes[:16]
    encrypted = encrypted_data_bytes[16:]
    cipher = Cipher(algorithms.AES(ENCRYPTION_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted) + decryptor.finalize()
    return decrypted.decode('utf-8').strip()

# App function to get AI response
@st.cache_data(ttl=3600)
def get_ai_response(prompt: str, email_content: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Features and Analysis Functions

# SLA Breach Detection
def detect_sla_breach(text: str) -> str:
    if "SLA" in text and ("breach" in text or "violation" in text):
        return "SLA Breach Detected: Immediate attention required."
    return "No SLA breach detected."

# Issue Prioritization
def prioritize_issues(text: str) -> str:
    if "urgent" in text or "critical" in text:
        return "Priority: High"
    return "Priority: Normal"

# Resolution Time Prediction (Dummy example)
def predict_resolution_time(text: str) -> str:
    if "complex" in text:
        return "Estimated Resolution Time: 48 hours"
    return "Estimated Resolution Time: 24 hours"

# Stakeholder Identification
def identify_stakeholders(text: str) -> List[str]:
    stakeholders = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", text)
    return stakeholders

# Escalation Level Definition
def define_escalation_level(text: str) -> str:
    if "critical" in text:
        return "Escalation Level: Level 3"
    elif "high" in text:
        return "Escalation Level: Level 2"
    return "Escalation Level: Level 1"

# Performance Metrics Extraction
def extract_performance_metrics(text: str) -> str:
    metrics = ["performance", "efficiency", "output", "target", "goal"]
    found_metrics = [metric for metric in metrics if metric in text.lower()]
    return ", ".join(found_metrics) if found_metrics else "No performance metrics found."

# Root Cause Detection
def detect_root_cause(text: str) -> str:
    if "system failure" in text:
        return "Root Cause: System failure"
    return "Root Cause: Not identified"

# Root Cause Frequency Analysis
def analyze_root_cause_frequency(text: str) -> str:
    root_causes = ["network issue", "software bug", "hardware failure"]
    cause_counts = {cause: text.lower().count(cause) for cause in root_causes}
    return json.dumps(cause_counts)

# Automated Response Suggestions
def generate_response(text: str) -> str:
    if "urgent" in text:
        return "Response: We are addressing the issue urgently and will update you shortly."
    return "Response: Thank you for your feedback, we are looking into it."

# Function to handle file upload and process content
def handle_uploaded_file(uploaded_file) -> str:
    try:
        if uploaded_file is not None:
            email_content = uploaded_file.read().decode("utf-8")
            return email_content
    except Exception as e:
        st.error(f"Error while reading file: {e}")
    return ""

# Main logic for processing email content
email_content = ""
if uploaded_file := st.file_uploader("Upload Email Content", type=["txt", "docx", "pdf"]):
    email_content = handle_uploaded_file(uploaded_file)

if email_content:
    try:
        # Encrypt sensitive email content
        encrypted_email = encrypt_data(email_content)
        decrypted_email = decrypt_data(encrypted_email)
        
        if features["sentiment"]:
            sentiment = get_ai_response("Sentiment analysis for this email: ", decrypted_email)
            st.subheader("Sentiment Analysis")
            st.write(sentiment)

        if features["highlights"]:
            highlights = get_ai_response("Highlight key phrases from this email: ", decrypted_email)
            st.subheader("Key Phrases Highlighted")
            st.write(highlights)
        
        if features["sla_breach"]:
            sla_breach = detect_sla_breach(decrypted_email)
            st.subheader("SLA Breach Detection")
            st.write(sla_breach)
        
        if features["prioritization"]:
            prioritization = prioritize_issues(decrypted_email)
            st.subheader("Issue Prioritization")
            st.write(prioritization)
        
        if features["resolution_time"]:
            resolution_time = predict_resolution_time(decrypted_email)
            st.subheader("Resolution Time Prediction")
            st.write(resolution_time)

        # Add additional features based on the selected features
        if features["stakeholder_identification"]:
            stakeholders = identify_stakeholders(decrypted_email)
            st.subheader("Stakeholders Identified")
            st.write(stakeholders)

        if features["escalation_levels"]:
            escalation_level = define_escalation_level(decrypted_email)
            st.subheader("Escalation Level")
            st.write(escalation_level)

        if features["performance_metrics"]:
            performance_metrics = extract_performance_metrics(decrypted_email)
            st.subheader("Performance Metrics")
            st.write(performance_metrics)

        if features["root_cause"]:
            root_cause = detect_root_cause(decrypted_email)
            st.subheader("Root Cause Detection")
            st.write(root_cause)
        
        if features["root_cause_frequency"]:
            root_cause_frequency = analyze_root_cause_frequency(decrypted_email)
            st.subheader("Root Cause Frequency")
            st.write(root_cause_frequency)

        if features["automated_response"]:
            automated_response = generate_response(decrypted_email)
            st.subheader("Automated Response Suggestion")
            st.write(automated_response)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload or paste email content to get insights.")

