from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# New: proper tabular preprocessing for prediction
from django.conf import settings
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# Create your views here.
from Remote_User.models import ClientRegister_Model,cardiac_arrest_prediction,detection_ratio,detection_accuracy


_PREDICT_PIPELINE = None
_PREDICT_CHOICES = None
_PREDICT_RANGES = None


def _get_predict_pipeline_and_choices():
    """
    Build (once) a proper ML pipeline using all relevant dataset features.
    Also returns dropdown choices derived from dataset unique categorical values.
    """
    global _PREDICT_PIPELINE, _PREDICT_CHOICES, _PREDICT_RANGES

    if _PREDICT_PIPELINE is not None and _PREDICT_CHOICES is not None and _PREDICT_RANGES is not None:
        return _PREDICT_PIPELINE, _PREDICT_CHOICES, _PREDICT_RANGES

    data_path = Path(settings.BASE_DIR) / "Datasets.csv"
    data = pd.read_csv(str(data_path), encoding="latin-1")

    # Target
    if "Results" in data.columns:
        y = pd.to_numeric(data["Results"], errors="coerce").fillna(0).astype(int)
    else:
        y = pd.to_numeric(data["HeartDisease"], errors="coerce").fillna(0).astype(int)

    # Features (exclude Fid: it's an identifier/noisy string in this dataset)
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numeric_cols = ["Age_In_Days", "RestingBP", "MaxHR", "Oldpeak", "slp", "caa", "thall"]
    feature_cols = categorical_cols + numeric_cols

    X = data[feature_cols].copy()
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(subset=numeric_cols)
    y = y.loc[X.index]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X, y)

    # Dropdown choices (dataset-driven).
    # For numeric cols we provide "dropdown suggestions" (datalist) and cap size.
    choices = {}
    for col in categorical_cols:
        choices[col] = sorted([str(v) for v in data[col].dropna().unique().tolist()])

    for col in numeric_cols:
        vals = pd.to_numeric(data[col], errors="coerce").dropna().unique().tolist()
        vals = sorted(vals)
        # cap to avoid huge dropdowns
        vals = vals[:250]
        choices[col] = [str(v).rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in vals]

    # Numeric ranges for UI constraints (use 1st-99th percentile to avoid outliers)
    ranges = {}
    for col in ["RestingBP", "MaxHR", "Oldpeak"]:
        s = pd.to_numeric(data[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        # make nicer numbers
        if col in ("RestingBP", "MaxHR"):
            lo = int(max(0, round(lo)))
            hi = int(max(lo + 1, round(hi)))
        else:  # Oldpeak can be float
            lo = round(lo, 2)
            hi = round(hi, 2)
        ranges[col] = {"min": lo, "max": hi}

    _PREDICT_PIPELINE = pipeline
    _PREDICT_CHOICES = choices
    _PREDICT_RANGES = ranges
    return pipeline, choices, ranges

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Cardiac_Arrest_Type(request):
    pipeline, choices, ranges = _get_predict_pipeline_and_choices()

    if request.method == "POST":
        Fid = request.POST.get('Fid', '')
        Age_In_Days = request.POST.get('Age_In_Days', '')
        Sex = request.POST.get('Sex', '')
        ChestPainType = request.POST.get('ChestPainType', '')
        RestingBP = request.POST.get('RestingBP', '')
        RestingECG = request.POST.get('RestingECG', '')
        MaxHR = request.POST.get('MaxHR', '')
        ExerciseAngina = request.POST.get('ExerciseAngina', '')
        Oldpeak = request.POST.get('Oldpeak', '')
        ST_Slope = request.POST.get('ST_Slope', '')
        slp = request.POST.get('slp', '')
        caa = request.POST.get('caa', '')
        thall = request.POST.get('thall', '')

        # Build 1-row dataframe for prediction (must match training columns)
        try:
            X_input = pd.DataFrame([{
                "Sex": Sex,
                "ChestPainType": ChestPainType,
                "RestingECG": RestingECG,
                "ExerciseAngina": ExerciseAngina,
                "ST_Slope": ST_Slope,
                "Age_In_Days": float(Age_In_Days),
                "RestingBP": float(RestingBP),
                "MaxHR": float(MaxHR),
                "Oldpeak": float(Oldpeak),
                "slp": float(slp),
                "caa": float(caa),
                "thall": float(thall),
            }])
        except Exception:
            return render(request, 'RUser/Predict_Cardiac_Arrest_Type.html', {
                'objs': "Please enter valid numeric values.",
                'choices': choices,
                'ranges': ranges,
            })

        prediction = int(pipeline.predict(X_input)[0])
        val = 'Cardiac Arrest Found' if prediction == 1 else 'No Cardiac Arrest Found'
        voice_text = "Sorry. Cardiac arrest found." if prediction == 1 else "Good news. No cardiac arrest found."

        # Suggested next steps (high-level; not a substitute for clinical judgement)
        if prediction == 1:
            urgency = "High"
            next_steps = [
                "Treat this as a HIGH-RISK alert. Do not rely on this result alone.",
                "Immediately notify the responsible clinician / NICU team and follow your hospital emergency protocol.",
                "Re-check and document vital signs and alarms (HR, SpO₂, BP, respiration) and verify sensor placement.",
                "Arrange prompt clinical assessment and confirmatory evaluation as per protocol (e.g., ECG/monitor review).",
                "If the baby is unresponsive or not breathing normally, activate emergency response per local guidelines.",
            ]
        else:
            urgency = "Low"
            next_steps = [
                "Continue routine monitoring as per NICU protocol.",
                "If symptoms/alarms occur, escalate to clinical review immediately.",
                "Consider repeating assessment with updated measurements if the condition changes.",
            ]
            positive_lines = [
                "Good news — no cardiac arrest is indicated by this assessment.",
                "Keep up the great care and continue regular monitoring.",
            ]
        disclaimer = "Note: This tool provides supportive information only and is NOT a medical decision system."

        cardiac_arrest_prediction.objects.create(
            Fid=Fid,
            Age_In_Days=Age_In_Days,
            Sex=Sex,
            ChestPainType=ChestPainType,
            RestingBP=RestingBP,
            RestingECG=RestingECG,
            MaxHR=MaxHR,
            ExerciseAngina=ExerciseAngina,
            Oldpeak=Oldpeak,
            ST_Slope=ST_Slope,
            slp=slp,
            caa=caa,
            thall=thall,
            Prediction=val
        )

        return render(request, 'RUser/Predict_Cardiac_Arrest_Type.html', {
            'objs': val,
            'choices': choices,
            'ranges': ranges,
            'urgency': urgency,
            'next_steps': next_steps,
            'positive_lines': positive_lines if prediction == 0 else None,
            'disclaimer': disclaimer,
            'voice_text': voice_text,
        })

    return render(request, 'RUser/Predict_Cardiac_Arrest_Type.html', {'choices': choices, 'ranges': ranges})



