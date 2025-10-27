import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

# ====== PAGE STYLE ======
st.set_page_config(page_title="🐾 VetCare AI System", layout="wide")
st.markdown("""
<style>
/* Remove top padding/margin inside Streamlit tabs */
div[data-testid="stTabs"] > div[role="tablist"] + div {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
section.main > div:first-child, .block-container {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stVerticalBlock"] > div { padding-top: 0 !important; }
[data-testid="stHorizontalBlock"] > div { padding-top: 0 !important; }

/* Remove margin/padding above and inside each card, especially first child of tab */
.glass-card {
    margin-top: 0 !important;
    padding-top: 1.5em !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
body {
  background: linear-gradient(133deg, #e0f7fa 0%, #fffde4 80%, #dbeafe 100%);
}
[data-testid="stAppViewContainer"], .main, .block-container {
  background: none !important;
  box-shadow: none !important;
}
.top-banner {
  width: 100%;
  margin-bottom: 2.3em;
  background: linear-gradient(105deg, #028090 0%, #f6d365 100%);
  box-shadow: 0 10px 30px #23a77111;
  padding: 2em 0 1.3em 0;
  border-radius: 0 0 40px 40px;
  text-align: center;
  letter-spacing: 0.05em;
  background-size: 150% 150%;
  animation: gradient-move 9s infinite alternate;
}
@keyframes gradient-move { 0%{background-position:0% 50%} 100%{background-position:100% 50%} }
.top-banner h1 { font-size: 2.6em; color: #fff; margin-bottom: 0.2em; font-family: 'Segoe UI', 'Roboto',sans-serif; font-weight: 700;}
.top-banner h4 {color: #224; font-size:1.19em; font-weight: 400; letter-spacing: 0.025em;}
.glass-card {background: rgba(237,252,255,0.71); border-radius: 32px; box-shadow: 0 6px 32px #02809022, 0 2px 6px #02809010; backdrop-filter: blur(2.2px); padding: 2.5em 2.3em 1.5em 2.3em; margin-bottom: 2em; border: 1.5px solid #b6fafa;}
.glass-card:hover {box-shadow:0 12px 46px #02809017;}
.stButton > button {
  color:#fff; border-radius: 26px;
  background: linear-gradient(90deg,#018ea1 10%,#67ecb8 100%);
  border:none;font-size:1.06em;font-weight:600;padding:0.67em 2.1em;
  margin-top:9px;box-shadow: 0 2px 12px #0e7d5920;transition: all 0.19s;}
.stButton > button:hover {background: linear-gradient(93deg,#3cb8a7 6%,#edee62 100%);}
.stTabs [role=tab] {font-size:1.12em;font-weight:700;}
#MainMenu, header, footer {visibility: hidden;}
a.custom-email { color:#00746f !important; text-decoration:none; font-weight:500; font-size:1.04em;}
a.custom-email:hover {text-decoration:underline;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
body { margin-top: 0 !important; }
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0rem !important;
    margin-top: 0 !important;
}
.block-container,
.main,
[data-testid="block-container"] {
    padding-top: 0rem !important;
    margin-top: 0 !important;
}
.css-18ni7ap,
.css-1dp5vir,
.css-12oz5g7 {
    padding-top: 0px !important;
    margin-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div class="top-banner">
    <h1>🐾 VetCare <span style="color:#ffe0a1;font-weight:500;">AI</span></h1>
    <h4>Personalized, professional animal care—delivered beautifully.</h4>
</div>
''', unsafe_allow_html=True)


# ====== MODEL LOADING ======
@st.cache_resource
def load_models():
    try:
        models = {}
        models['pregnancy'] = joblib.load("pregnancy_model_xgb.pkl")
        models['delivery'] = joblib.load("delivery_model_xgb.pkl")
        models['disease'] = joblib.load("disease_model_xgb.pkl")
        models['disease_encoder'] = joblib.load("disease_encoder.pkl")
        
        encoders = {}
        encoder_files = ['Species_encoder.pkl', 'Breed_encoder.pkl', 'Discharge_Type_encoder.pkl']
        for file in encoder_files:
            if os.path.exists(file):
                key = file.replace('_encoder.pkl', '')
                encoders[key] = joblib.load(file)
        return models, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run the training script first to generate model files.")
        return None, None

models, encoders = load_models()
if models is None:
    st.stop()
st.success("✅ All models loaded successfully!")

# ====== SIDEBAR ======
with st.sidebar:
    st.image("/Users/sathwiknomula/Downloads/Vet_animals.png", width=280)
    st.markdown("### 👩‍⚕️ Contact")
    st.markdown(
        '<a href="mailto:vethelplineindia@gmail.com" class="custom-email">📧 vethelplineindia@gmail.com</a>',
        unsafe_allow_html=True
    )
    st.markdown("- **24/7 Helpline:** +91-361-2651593")
    st.markdown("---")
    st.info("Always consult a certified veterinarian for any medical emergencies.")

# ====== APP TABS ======
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Pregnancy Prediction", 
    "📅 Delivery Estimation", 
    "🦠 Disease Detection", 
    "💬 VetBot Assistant"
])


# ---- PREGNANCY PREDICTION ----
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🔍 Pregnancy Prediction")
    
    # Basic animal information
    col1, col2 = st.columns(2)
    with col1:
        species = st.selectbox("Species", ["Dog", "Cat", "Cow", "Goat", "Horse", "Sheep"], key="preg_species")
        breed = st.text_input("Breed", placeholder="e.g., German Shepherd", key="preg_breed")
        age = st.number_input("Age (Years)", min_value=0.0, max_value=20.0, value=3.0, step=0.1, key="preg_age")
        
    with col2:
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=1000.0, value=25.0, step=0.1, key="preg_weight")
        body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=110.0, value=101.5, step=0.1, key="preg_temp")
    
    st.markdown("---")
    st.subheader("🎯 Primary Pregnancy Indicators")
    
    # Main prediction features - FIXED VARIABLE NAMES
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**🫀 Fetal Heart Sound Detection**")
        fetal_sound = st.radio(
            "Can you detect fetal heart sounds?",
            options=["No", "Yes"],
            key="preg_fetal",
            help="Fetal heart sounds are typically detectable after 25-30 days of pregnancy"
        )
        
        if fetal_sound == "Yes":
            st.success("✅ Fetal heart sounds detected")
        else:
            st.info("❌ No fetal heart sounds detected")
        
    with col4:
        st.markdown("**🐕 Behavioral Changes**")
        behavior_change = st.radio(
            "Are there notable behavioral changes?",
            options=["No", "Yes"],
            key="preg_behavior",
            help="Changes in nesting behavior, appetite, activity level, or affection"
        )
        
        if behavior_change == "Yes":
            st.success("✅ Behavioral changes observed")
        else:
            st.info("❌ No significant behavioral changes")
    
    if st.button("🔍 Predict Pregnancy", type="primary", key="predict_preg_btn"):
        try:
            # ============ SIMPLIFIED PREGNANCY LOGIC (FIXED) ============
            
            # Check both conditions using CORRECT variable names
            fetal_detected = (fetal_sound == "Yes")
            behavior_detected = (behavior_change == "Yes")  # FIXED: was "behavior_changed"
            
            # Pregnancy prediction logic
            if fetal_detected and behavior_detected:
                # Both primary indicators positive
                prediction = "PREGNANT"
                confidence = 95
                status_color = "success"
                icon = "🤰"
                explanation = "Both primary pregnancy indicators are positive"
                recommendations = [
                    "Schedule regular veterinary checkups",
                    "Upgrade to pregnancy-specific nutrition",
                    "Prepare birthing area and supplies",
                    "Monitor for delivery signs"
                ]
                
            elif fetal_detected and not behavior_detected:
                # Only fetal heart sound detected
                prediction = "LIKELY PREGNANT"
                confidence = 80
                status_color = "warning"
                icon = "⚠️"
                explanation = "Fetal heart sounds detected but behavioral changes not yet apparent"
                recommendations = [
                    "Continue monitoring for behavioral changes",
                    "Schedule veterinary confirmation",
                    "Begin prenatal care preparations"
                ]
                
            elif not fetal_detected and behavior_detected:
                # Only behavior changes detected
                prediction = "POSSIBLY PREGNANT - EARLY STAGE"
                confidence = 40
                status_color = "warning"
                icon = "🤔"
                explanation = "Behavioral changes present but no fetal heart sounds yet detected"
                recommendations = [
                    "Re-examine in 7-14 days for fetal heart sounds",
                    "Consider hormonal testing (Relaxin)",
                    "Schedule veterinary consultation"
                ]
                
            else:
                # Neither primary indicator positive - FIXED LOGIC
                prediction = "NOT PREGNANT"
                confidence = 90
                status_color = "error"
                icon = "❌"
                explanation = "No primary pregnancy indicators detected"
                recommendations = [
                    "If breeding was recent, re-examine in 2-3 weeks",
                    "Consider alternative causes for any symptoms",
                    "Review breeding timing and methods"
                ]
            
            # ============ DISPLAY RESULTS ============
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Main prediction display
            if status_color == "success":
                st.success(f"{icon} **{prediction}**")
            elif status_color == "warning":
                st.warning(f"{icon} **{prediction}**")
            else:
                st.error(f"{icon} **{prediction}**")
            
            # Confidence metric
            st.metric("Confidence Level", f"{confidence}%")
            
            # Clinical explanation
            st.info(f"**Clinical Analysis:** {explanation}")
            
            # Recommendations
            st.markdown("---")
            st.subheader("📋 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Assessment summary
            st.markdown("---")
            st.subheader("📊 Assessment Summary")
            col7, col8 = st.columns(2)
            with col7:
                st.metric("Fetal Heart Sound", "✅ Detected" if fetal_detected else "❌ Not Detected")
            with col8:
                st.metric("Behavior Changes", "✅ Present" if behavior_detected else "❌ Absent")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check your inputs and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---- DELIVERY ESTIMATION ----
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📅 Delivery Date Estimation")
    col1, col2 = st.columns(2)
    with col1:
        mating_date = st.date_input("Mating Date", value=datetime.now().date() - timedelta(days=30))
        species_delivery = st.selectbox("Species", ["Dog", "Cat", "Cow", "Goat", "Horse", "Sheep", "Pig"], key="delivery_species")
    with col2:
        current_weight = st.number_input("Current Weight (kg)", min_value=0.0, value=25.0, key="delivery_weight")
        current_temp = st.number_input("Current Temperature (°F)", min_value=95.0, max_value=110.0, value=101.5, key="delivery_temp")
    if st.button("📅 Estimate Delivery Date", type="primary"):
        gestation_days = {
            "Dog": 63, "Cat": 65, "Cow": 283, "Goat": 150,
            "Horse": 342, "Sheep": 147, "Pig": 114
        }
        base_days = gestation_days.get(species_delivery, 63)
        try:
            if current_weight and current_temp:
                model_data = pd.DataFrame({
                    'Body_Temperature_F': [current_temp],
                    'Weight_kg': [current_weight]
                })
                ai_days = models['delivery'].predict(model_data)[0]
                estimated_days = int((base_days + ai_days) / 2)
            else:
                estimated_days = base_days
        except:
            estimated_days = base_days
        delivery_date = mating_date + timedelta(days=estimated_days)
        days_remaining = (delivery_date - datetime.now().date()).days
        st.success(f"🗓️ **Estimated Delivery Date:** {delivery_date.strftime('%B %d, %Y')}")
        if days_remaining > 0:
            st.info(f"⏰ **Days Remaining:** {days_remaining} days")
            progress = min(100, ((estimated_days - days_remaining) / estimated_days) * 100)
            st.progress(progress / 100)
            if progress < 30:
                stage = "Early Pregnancy"
                advice = "Focus on nutrition and regular checkups"
            elif progress < 70:
                stage = "Mid Pregnancy"
                advice = "Monitor weight gain and prepare birthing area"
            else:
                stage = "Late Pregnancy"
                advice = "Watch for signs of labor and stay close to vet"
            st.write(f"**Stage:** {stage} ({progress:.0f}% complete)")
            st.write(f"**Advice:** {advice}")
        elif days_remaining == 0:
            st.warning("🚨 **Due Today!** Monitor closely for labor signs.")
        else:
            st.error(f"⚠️ **Overdue by {abs(days_remaining)} days!** Contact veterinarian immediately.")
    st.markdown('</div>', unsafe_allow_html=True)


# ---- DISEASE DETECTION ----
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🦠 Disease Detection")
    c1, c2 = st.columns(2)
    
    with c1:
        disease_species = st.selectbox("Species", ["Dog", "Cat", "Cow", "Goat", "Horse"], key="disease_species")
        disease_breed = st.text_input("Breed", key="disease_breed")
        disease_age = st.number_input("Age (Years)", min_value=0.0, max_value=20.0, value=3.0, key="disease_age")
        disease_weight = st.number_input("Weight (kg)", min_value=0.0, value=25.0, key="disease_weight")
        
    with c2:
        st.markdown("**🎯 Disease Detection Criteria**")
        brucella_result = st.selectbox("Brucellosis Test", ["Negative", "Positive"], key="disease_brucella")
        toxoplasma_result = st.selectbox("Toxoplasmosis Test", ["Negative", "Positive"], key="disease_toxo")
        disease_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=110.0, value=101.5, key="disease_temp")
        lethargy = st.selectbox("Lethargy/Weakness", ["No", "Yes"], key="lethargy3")
            
    if st.button("🔬 Analyze Disease Risk", type="primary"):
        try:
            # ============ STRICT 3-CONDITION LOGIC ============
            
            # Condition 1: Test positive
            test_positive = (brucella_result == "Positive" or toxoplasma_result == "Positive")
            
            # Condition 2: Fever >102°F  
            fever_present = (disease_temp > 102.0)
            
            # Condition 3: Lethargy = Yes
            lethargy_present = (lethargy == "Yes")
            
            # ============ DISEASE DETECTION LOGIC ============
            if test_positive and fever_present and lethargy_present:
                # ALL 3 CONDITIONS MET - DISEASE DETECTED
                if brucella_result == "Positive" and toxoplasma_result == "Positive":
                    disease_prediction = "BRUCELLOSIS + TOXOPLASMOSIS (Co-infection)"
                    confidence = 0.95
                    icon = "🚨"
                elif brucella_result == "Positive":
                    disease_prediction = "BRUCELLOSIS DETECTED"
                    confidence = 0.92
                    icon = "🦠"
                else:
                    disease_prediction = "TOXOPLASMOSIS DETECTED"
                    confidence = 0.90
                    icon = "🧠"
                    
                st.error(f"{icon} **DISEASE DETECTED:** {disease_prediction}")
                st.metric("Diagnostic Confidence", f"{confidence:.0%}")
                
                # ============ DISEASE DESCRIPTIONS & ACTIONS ============
                if "BRUCELLOSIS" in disease_prediction:
                    st.info("ℹ️ **Brucellosis** is a highly contagious bacterial infection causing reproductive failures, abortions, and poses serious zoonotic risk to humans.")
                    st.error("🚨 **BRUCELLOSIS - IMMEDIATE ACTIONS REQUIRED:**")
                    st.write("1. **Isolate animal immediately** from all other animals")
                    st.write("2. **Contact veterinarian urgently** - reportable disease")
                    st.write("3. **Use protective equipment** - gloves, masks when handling")
                    st.write("4. **Test all contact animals** for infection")
                    st.write("5. **Report to authorities** - notify local veterinary department")
                    
                elif "TOXOPLASMOSIS" in disease_prediction:
                    st.info("ℹ️ **Toxoplasmosis** is a parasitic infection affecting the nervous system, causing neurological symptoms and reproductive problems in animals.")
                    st.warning("⚠️ **TOXOPLASMOSIS - VETERINARY CONSULTATION REQUIRED:**")
                    st.write("1. **Schedule emergency vet appointment** within 24-48 hours")
                    st.write("2. **Isolate from pregnant animals** - abortion risk")
                    st.write("3. **Monitor neurological symptoms** - seizures, coordination")
                    st.write("4. **Start antibiotic treatment** as prescribed by veterinarian")
                    st.write("5. **Improve facility hygiene** - clean water, proper sanitation")
                
                # Co-infection protocol
                if brucella_result == "Positive" and toxoplasma_result == "Positive":
                    st.error("🚨 **CRITICAL CO-INFECTION - EMERGENCY PROTOCOL:**")
                    st.write("1. **Maximum isolation** - complete facility lockdown")
                    st.write("2. **Immediate veterinary specialist** consultation required")
                    st.write("3. **Enhanced protective equipment** mandatory for all staff")
                    st.write("4. **Government notification** - disease outbreak reporting")
                    st.write("5. **Consider euthanasia** - extremely high zoonotic risk")
                
            else:
                # NOT ALL CONDITIONS MET - NO DISEASE DETECTED
                st.success("✅ **NO DISEASE DETECTED**")
                st.metric("Diagnostic Confidence", "85%")
                
                # ============ ROUTINE MONITORING ACTIONS ============
                st.info("💡 **ROUTINE HEALTH MONITORING:**")
                st.write("1. **Continue daily health checks** - temperature, appetite, behavior")
                st.write("2. **Maintain vaccination schedule** - follow veterinary guidelines")
                st.write("3. **Schedule routine checkups** - every 6 months")
                st.write("4. **Monitor for symptom changes** - watch for fever or lethargy")
                st.write("5. **Keep detailed health records** - document all observations")
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- VETBOT TAB ----
def vetbot_response(query):
    query = query.lower()
    responses = {
        "pregnancy_care": "🔍 **Pregnancy Care Tips:**\n- Provide high-quality protein and calcium\n- Ensure clean, quiet environment\n- Regular vet checkups\n- Monitor for unusual discharge or behavior",
        "delivery_signs": "🚨 **Signs of Labor:**\n- Restlessness and nesting behavior\n- Drop in body temperature\n- Loss of appetite\n- Contractions and pushing\n- Clear or bloody discharge",
        "brucellosis": "⚠️ **Brucellosis Alert:**\n- Highly contagious bacterial infection\n- Causes abortion in pregnant animals\n- Isolate immediately and contact vet\n- Symptoms: fever, lethargy, abnormal discharge",
        "toxoplasmosis": "🦠 **Toxoplasmosis Info:**\n- Parasitic infection dangerous for pregnant animals\n- Can cause stillbirths and neurological issues\n- Symptoms: loss of appetite, diarrhea, fever\n- Prevent through proper hygiene",
        "nutrition": "🥗 **Pregnancy Nutrition:**\n- Increase protein by 25-50%\n- Add calcium and phosphorus supplements\n- Feed smaller, frequent meals\n- Provide fresh water always",
        "emergency": "🚨 **Emergency Signs:**\n- Prolonged labor (>2 hours active pushing)\n- Green/black discharge before birth\n- Severe lethargy or collapse\n- High fever (>103°F)\n- **Call vet immediately!**",
        "pregnancy_prediction": "🔮 **Pregnancy Prediction Info:**\n- Dogs: confirm via ultrasound ~20–25 days\n- Cats: confirm via ultrasound ~15–20 days\n- Cows: ultrasound ~25–30 days\n- Goats/Sheep: ~25–30 days\n- Horses: ultrasound ~14–16 days\n- Pigs: ~22–25 days",
        "delivery_estimation": "🗓️ **Delivery Estimation:**\n- Dog: ~63 days from mating\n- Cat: ~65 days\n- Cow: ~283 days\n- Goat: ~150 days\n- Sheep: ~147 days\n- Horse: ~342 days\n- Pig: ~114 days",
        "disease_detection": "🧬 **General Disease Detection:**\n- Observe appetite, body temperature, and behavior changes\n- Common pregnancy-related diseases: Brucellosis, Toxoplasmosis, Pyometra\n- Diagnostic methods: blood test, ultrasound, cultures\n- Consult a vet for accurate confirmation",
        "vaccination": "💉 **Vaccination Guidance for Pregnant Animals:**\n- Avoid live vaccines during pregnancy\n- Core vaccines should be given before breeding\n- Deworming and parasite control are essential\n- Consult a vet for species-specific schedules",
        "post_delivery": "🤱 **Post-Delivery Care:**\n- Ensure newborns are breathing and nursing\n- Provide warmth and clean bedding\n- Monitor mother for retained placenta or fever\n- Offer nutrient-rich food and fresh water"
    }

    delivery_days = {
        "dog": 63, "cat": 65, "cow": 283, "goat": 150,
        "sheep": 147, "horse": 342, "pig": 114
    }

    # Query mapping
    if any(word in query for word in ["pregnant", "pregnancy"]) and any(word in query for word in ["care", "feed", "diet"]):
        return responses["pregnancy_care"]
    elif any(word in query for word in ["delivery", "labor", "birth", "signs"]):
        return responses["delivery_signs"]
    elif "brucell" in query:
        return responses["brucellosis"]
    elif "toxoplasm" in query:
        return responses["toxoplasmosis"]
    elif any(word in query for word in ["nutrition", "diet", "feed"]):
        return responses["nutrition"]
    elif any(word in query for word in ["emergency", "urgent", "help"]):
        return responses["emergency"]
    elif any(word in query for word in ["predict", "prediction", "confirm pregnancy", "preg test"]):
        return responses["pregnancy_prediction"]
    elif any(word in query for word in ["delivery date", "expected date", "due date", "when will", "gestation length"]):
        return responses["delivery_estimation"]
    elif any(word in query for word in ["disease", "infection", "illness", "symptom"]):
        return responses["disease_detection"]
    elif any(word in query for word in ["vaccine", "vaccination", "immunization"]):
        return responses["vaccination"]
    elif any(word in query for word in ["post delivery", "after birth", "mother care", "lactation"]):
        return responses["post_delivery"]
    elif "how many days" in query or "gestation" in query:
        for animal, days in delivery_days.items():
            if animal in query:
                return f"🗓️ **{animal.title()} Gestation Period:** {days} days"
        return "🗓️ **Typical Gestation Periods:**\n" + "\n".join([f"- {animal.title()}: {days} days" for animal, days in delivery_days.items()])
    
    return "🤖 **VetBot:** I'm here to help with pregnancy care, prediction, delivery, diseases, and nutrition. Try asking about:\n- Pregnancy prediction\n- Delivery estimation\n- Disease detection\n- Vaccination guidance\n- Emergency or post-delivery care"

with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("💬 VetBot - Your Veterinary Assistant")
    st.subheader("💡 Try asking about:")
    sample_questions = [
        "How to care for a pregnant dog?",
        "What are signs of delivery?",
        "How many days does a cow pregnancy last?",
        "When will my goat deliver?",
        "How to predict if my cat is pregnant?",
        "Tell me about brucellosis",
        "What diseases affect pregnant animals?",
        "Emergency signs during pregnancy",
        "Nutrition tips for pregnant animals",
        "What vaccines are safe for pregnant cows?",
        "How to care for mother after delivery?"
    ]
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"💬 {question}", key=f"sample_{i}"):
                st.session_state.user_query = question

    user_query = st.text_input(
        "Ask me anything about veterinary care:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., When will my goat deliver?"
    )
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        response = vetbot_response(user_query)
        with st.chat_message("assistant"):
            st.markdown(response)
        if 'user_query' in st.session_state:
            del st.session_state.user_query
    st.markdown('</div>', unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""<div style="text-align:center;color:#8d9989;font-size:1em;margin-top:2.5em;">
© 2025 VetCare AI. Designed for animal health professionals and pet lovers worldwide.
</div>
""", unsafe_allow_html=True)
