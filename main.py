import os
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
app = FastAPI()
load_dotenv()

# Initialize the LLM (e.g., Google Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Define API URLs from environment variables
MALE_API_URL = os.getenv("MALE_BN_API_URL", "https://male-bn-api.com/predict")
FEMALE_API_URL = os.getenv("FEMALE_BN_API_URL", "https://female-bn-api.com/predict")

# Helper function to get risk probabilities from API
def parse_probability(prob_str):
    return float(prob_str.strip('%')) / 100

# Helper function to get risk probabilities from API
def get_risk_probabilities(patient_data):
    
    payload = patient_data.copy()
    payload.pop('gender', None)
    if patient_data["gender"] == "M":
        
        api_url = MALE_API_URL
    elif patient_data["gender"] == "F":
        api_url = FEMALE_API_URL
    else:
        raise ValueError("Invalid gender")
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    return response.json()

# Define structured output for LLM recommendations
class Recommendations(BaseModel):
    patient_recommendations: list[str]
    doctor_recommendations: list[str]

# Define the state structure
class State(TypedDict):
    patient_data: dict
    risk_probabilities: dict
    recommendations: Recommendations
    selected_patient_recommendations: list[str]

# Node 1: Calculate risk probabilities using API
def risk_assessment(state: State):
    patient_data = state["patient_data"]
    probabilities = get_risk_probabilities(patient_data)
    return {"risk_probabilities": probabilities}

# Node 2: Generate recommendations using LLM
def generate_recommendations(state: State):
    
    patient_data = state["patient_data"]
    risk_probabilities = state["risk_probabilities"]
    
    prompt = f"""
    Based on this patient data and risk probabilities, provide up to five recommendations: 
    at least three for the patient (lifestyle changes) and two for the doctor (medical actions).
    
    Patient Data: {patient_data}
    Diabetes Probability: {risk_probabilities['Health Risk Probabilities']['Diabetes']}
    CVD Probability: {risk_probabilities['Health Risk Probabilities']['Heart Disease']}
    
    Ensure recommendations are diverse, personalized, and specific to the patient's profile.
    Output as a JSON object with 'patient_recommendations' and 'doctor_recommendations' lists.
    """
    
    recommendations = llm.with_structured_output(Recommendations).invoke(prompt)
   
    return {"recommendations": recommendations}

# Node 3: Evaluate patient recommendations by simulating impact
def evaluate_recommendations(state: State):
    patient_data = state["patient_data"]
    risk_probabilities = state["risk_probabilities"]
    recommendations = state["recommendations"]
    
    selected_patient_recommendations = []
    
    for rec in recommendations.patient_recommendations:
        rec_type = classify_recommendation(rec)
        if rec_type != "Other":
            adjusted_data = adjust_metrics(patient_data, rec_type)
            new_probs = get_risk_probabilities(adjusted_data)
            if is_effective(risk_probabilities, new_probs):
                selected_patient_recommendations.append(rec)
    
    return {"selected_patient_recommendations": selected_patient_recommendations}

# Helper: Classify recommendation type based on keywords
def classify_recommendation(text: str) -> str:
    text = text.lower()
    if "exercise" in text or "physical activity" in text:
        return "Physical Activity"
    elif "diet" in text or "nutrition" in text or "eat" in text:
        return "Diet"
    elif "smoking" in text or "quit" in text:
        return "Smoking Cessation"
    else:
        return "Other"

# Helper: Adjust patient data based on recommendation type
def adjust_metrics(patient_data: dict, rec_type: str) -> dict:
    adjusted_data = patient_data.copy()
    if rec_type == "Physical Activity":
        
        adjusted_data["Exercise_Hours_Per_Week"] += 2
        # Only adjust if key exists to avoid KeyError
        if "Physical_Activity_Days_Per_Week" in adjusted_data:
            adjusted_data["Physical_Activity_Days_Per_Week"] += 1
    elif rec_type == "Diet":
        adjusted_data["BMI"] -= 1
        adjusted_data["glucose"] -= 10
    elif rec_type == "Smoking Cessation":
        adjusted_data["is_smoking"] = 0
    return adjusted_data

# Helper: Check if recommendation reduces risk significantly
def is_effective(original_probs: dict, new_probs: dict) -> bool:
    orig_diabetes = parse_probability(original_probs['Health Risk Probabilities']['Diabetes'])
    orig_cvd = parse_probability(original_probs['Health Risk Probabilities']['Heart Disease'])
    new_diabetes = parse_probability(new_probs['Health Risk Probabilities']['Diabetes'])
    new_cvd = parse_probability(new_probs['Health Risk Probabilities']['Heart Disease'])
    
    # Accept if one risk drops by 0.05 and the other doesn't increase much
    return (new_diabetes < orig_diabetes - 0.05 and new_cvd <= orig_cvd + 0.01) or \
           (new_cvd < orig_cvd - 0.05 and new_diabetes <= orig_diabetes + 0.01)

# Node 4: Format the final output
def output_results(state: State):
    risk_probabilities = state["risk_probabilities"]
    selected_patient_recommendations = state["selected_patient_recommendations"]
    doctor_recommendations = state["recommendations"].doctor_recommendations[:2]  # Limit to 2
    
    return {
        "diabetes_probability": risk_probabilities['Health Risk Probabilities']['Diabetes'],
        "cvd_probability": risk_probabilities['Health Risk Probabilities']['Heart Disease'],
        "patient_recommendations": selected_patient_recommendations[:3],  # Limit to 3
        "doctor_recommendations": doctor_recommendations
    }

# Build the LangGraph workflow
graph_builder = StateGraph(State)
graph_builder.add_node("risk_assessment", risk_assessment)
graph_builder.add_node("generate_recommendations", generate_recommendations)
graph_builder.add_node("evaluate_recommendations", evaluate_recommendations)
graph_builder.add_node("output_results", output_results)

graph_builder.add_edge(START, "risk_assessment")
graph_builder.add_edge("risk_assessment", "generate_recommendations")
graph_builder.add_edge("generate_recommendations", "evaluate_recommendations")
graph_builder.add_edge("evaluate_recommendations", "output_results")
graph_builder.add_edge("output_results", END)

graph = graph_builder.compile()

# # Your provided patient data
# patient_data = {
#   "Blood_Pressure": 98.5,
#   "Age": 64,
#   "Exercise_Hours_Per_Week": 19,
#   "Diet": "Unhealthy",
#   "Sleep_Hours_Per_Day": 9,
#   "Stress_Level": 5,
#   "glucose": 129,
#   "BMI": 27,
#   "hypertension": 1,
#   "is_smoking": 1,
#   "hemoglobin_a1c":5.2,
#   "Diabetes_pedigree": 0,
#   "CVD_Family_History": 0,
#   "ld_value": 169,
#   "admission_tsh": 0.56,
#   "is_alcohol_user": 0,
#   "creatine_kinase_ck": 238,
#   'gender':'M',
# }

# # Run the AI agent
# initial_state = {"patient_data": patient_data}
# result = graph.invoke(initial_state)
# print(result)
# if result["selected_patient_recommendations"]:
#     print("Last Patient Recommendation:", result["selected_patient_recommendations"][-1])
# else:
#     print("No Patient Recommendations")
# if result["recommendations"].doctor_recommendations:
#     print("Last Doctor Recommendation:", result["recommendations"].doctor_recommendations[-1])
# else:
#     print("No Doctor Recommendations")



class PatientData(BaseModel):
    Blood_Pressure: float
    Age: int
    Exercise_Hours_Per_Week: int
    Diet: str
    Sleep_Hours_Per_Day: int
    Stress_Level: int
    glucose: int
    BMI: float
    hypertension: int
    is_smoking: int
    hemoglobin_a1c: float
    Diabetes_pedigree: int
    CVD_Family_History: int
    ld_value: int
    admission_tsh: float
    is_alcohol_user: int
    creatine_kinase_ck: int
    gender: str


@app.post("/predict")
async def predict(patient_data: PatientData):
    try:
        initial_state = {"patient_data": patient_data.dict()}
        result = graph.invoke(initial_state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "AI Agent is running"}