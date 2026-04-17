import requests

from config.api_config import API_URL

def predict_from_api(age, gender, edu, income, emp, home, amnt, intent, rate, percent, cred_len, score):
    payload = {
        "person_age": age,
        "person_gender": gender,
        "person_education": edu,
        "person_income": income,
        "person_emp_exp": emp,
        "person_home_ownership": home,
        "loan_amnt": amnt,
        "loan_intent": intent,
        "loan_int_rate": rate,
        "loan_percent_income": percent,
        "cb_person_cred_hist_length": cred_len,
        "credit_score": score
    }

    res = requests.post(API_URL, json=payload)
    res.raise_for_status()

    return res.json()
