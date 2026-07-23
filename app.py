"""
=========================================================
Employee Attrition Prediction Using ANN
Flask Application
=========================================================
"""

import os
import warnings
import logging
import traceback

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)

from tensorflow.keras.models import load_model

from config import *

from database import (
    db,
    init_db,
    PredictionHistory
)

warnings.filterwarnings("ignore")

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

# ==========================================================
# FLASK
# ==========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = SECRET_KEY

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

init_db(app)

# ==========================================================
# LOAD MODEL
# ==========================================================

logging.info(
    "Loading ANN model..."
)

if not os.path.exists(
    ANN_MODEL_PATH
):

    raise FileNotFoundError(

        "ann_model.keras tidak ditemukan."

    )

model = load_model(

    ANN_MODEL_PATH

)

logging.info(
    "ANN loaded."
)

# ==========================================================
# LOAD PREPROCESSOR
# ==========================================================

logging.info(
    "Loading preprocessor..."
)

if not os.path.exists(
    PREPROCESSOR_PATH
):

    raise FileNotFoundError(

        "preprocessor.pkl tidak ditemukan."

    )

preprocessor = joblib.load(

    PREPROCESSOR_PATH

)

logging.info(
    "Preprocessor loaded."
)

# ==========================================================
# LOAD FEATURE COLUMNS
# ==========================================================

logging.info(
    "Loading feature columns..."
)

if not os.path.exists(
    FEATURE_COLUMNS_PATH
):

    raise FileNotFoundError(

        "feature_columns.pkl tidak ditemukan."

    )

feature_columns = joblib.load(

    FEATURE_COLUMNS_PATH

)

logging.info(
    f"{len(feature_columns)} features loaded."
)

# ==========================================================
# SAFE INTEGER
# ==========================================================

def safe_int(value):

    try:

        return int(value)

    except:

        return 0


# ==========================================================
# SAFE FLOAT
# ==========================================================

def safe_float(value):

    try:

        return float(value)

    except:

        return 0.0

# ==========================================================
# READ FORM
# ==========================================================

def get_form_data():

    form = {

        "Age":
            safe_int(
                request.form.get("age")
            ),

        "Gender":
            request.form.get("gender"),

        "Marital_Status":
            request.form.get("marital_status"),

        "Department":
            request.form.get("department"),

        "Job_Role":
            request.form.get("job_role"),

        "Job_Level":
            safe_int(
                request.form.get("job_level")
            ),

        "Monthly_Income":
            safe_float(
                request.form.get("monthly_income")
            ),

        "Hourly_Rate":
            safe_float(
                request.form.get("hourly_rate")
            ),

        "Years_at_Company":
            safe_int(
                request.form.get("years_at_company")
            ),

        "Years_In_Current_Role":
            safe_int(
                request.form.get("years_in_current_role")
            ),

        "Years_Since_Last_Promotion":
            safe_int(
                request.form.get("years_since_last_promotion")
            ),

        "Work_Life_Balance":
            safe_int(
                request.form.get("work_life_balance")
            ),

        "Job_Satisfaction":
            safe_int(
                request.form.get("job_satisfaction")
            ),

        "Performance_Rating":
            safe_int(
                request.form.get("performance_rating")
            ),

        "Training_Hours_Last_Year":
            safe_int(
                request.form.get("training_hours_last_year")
            ),

        "Overtime":
            request.form.get("overtime"),

        "Project_Count":
            safe_int(
                request.form.get("project_count")
            ),

        "Average_Hours_Worked_Per_Week":
            safe_float(
                request.form.get(
                    "average_hours_worked_per_week"
                )
            ),

        "Absenteeism":
            safe_float(
                request.form.get(
                    "absenteeism"
                )
            ),

        "Work_Environment_Satisfaction":
            safe_int(
                request.form.get(
                    "work_environment_satisfaction"
                )
            ),

        "Relationship_With_Manager":
            safe_int(
                request.form.get(
                    "relationship_with_manager"
                )
            ),

        "Job_Involvement":
            safe_int(
                request.form.get(
                    "job_involvement"
                )
            ),

        "Distance_From_Home":
            safe_float(
                request.form.get(
                    "distance_from_home"
                )
            ),

        "Number_Of_Companies_Worked":
            safe_int(
                request.form.get(
                    "number_of_companies_worked"
                )
            )

    }

    return form

# ==========================================================
# PREDICT
# ==========================================================

def predict_employee(data):

    dataframe = pd.DataFrame(

        [data]

    )

    transformed = preprocessor.transform(

        dataframe

    )

    probability = float(

        model.predict(

            transformed,

            verbose=0

        )[0][0]

    )

    prediction = "Leave"

    if probability < 0.5:

        prediction = "Stay"

    return (

        prediction,

        probability

    )

# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ==========================================================
# PREDICT
# ==========================================================

@app.route(

    "/predict",

    methods=["POST"]

)

def predict():

    try:

        # ---------------------------------------------
        # Read Form
        # ---------------------------------------------

        form_data = get_form_data()

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        prediction, probability = predict_employee(

            form_data

        )

        # ---------------------------------------------
        # Save Database
        # ---------------------------------------------

        history = PredictionHistory(

            age=form_data["Age"],

            gender=form_data["Gender"],

            marital_status=form_data["Marital_Status"],

            department=form_data["Department"],

            job_role=form_data["Job_Role"],

            job_level=form_data["Job_Level"],

            monthly_income=form_data["Monthly_Income"],

            hourly_rate=form_data["Hourly_Rate"],

            years_at_company=form_data["Years_at_Company"],

            years_in_current_role=form_data["Years_In_Current_Role"],

            years_since_last_promotion=form_data[
                "Years_Since_Last_Promotion"
            ],

            work_life_balance=form_data[
                "Work_Life_Balance"
            ],

            job_satisfaction=form_data[
                "Job_Satisfaction"
            ],

            performance_rating=form_data[
                "Performance_Rating"
            ],

            training_hours_last_year=form_data[
                "Training_Hours_Last_Year"
            ],

            overtime=form_data[
                "Overtime"
            ],

            project_count=form_data[
                "Project_Count"
            ],

            average_hours_worked_per_week=form_data[
                "Average_Hours_Worked_Per_Week"
            ],

            absenteeism=form_data[
                "Absenteeism"
            ],

            work_environment_satisfaction=form_data[
                "Work_Environment_Satisfaction"
            ],

            relationship_with_manager=form_data[
                "Relationship_With_Manager"
            ],

            job_involvement=form_data[
                "Job_Involvement"
            ],

            distance_from_home=form_data[
                "Distance_From_Home"
            ],

            number_of_companies_worked=form_data[
                "Number_Of_Companies_Worked"
            ],

            prediction=prediction,

            probability=round(

                probability * 100,

                2

            )

        )

        db.session.add(

            history

        )

        db.session.commit()

        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        return render_template(

            "result.html",

            prediction=prediction,

            probability=round(

                probability * 100,

                2

            ),

            data=form_data

        )

    except Exception as error:

        db.session.rollback()

        logging.exception(

            error

        )

        flash(

            str(error),

            "danger"

        )

        return redirect(

            url_for(

                "index"

            )

        )

# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
def dashboard():

    total_prediction = PredictionHistory.query.count()

    stay_count = PredictionHistory.query.filter_by(
        prediction="Stay"
    ).count()

    leave_count = PredictionHistory.query.filter_by(
        prediction="Leave"
    ).count()

    latest_prediction = PredictionHistory.query.order_by(
        PredictionHistory.created_at.desc()
    ).limit(5).all()

    metrics = {
        "accuracy": 0.6380,
        "precision": 0.1881,
        "recall": 0.2456,
        "f1": 0.2130,
        "roc": 0.5082,
        "report": open(
            "reports/evaluation.txt",
            encoding="utf-8"
        ).read()
    }

    return render_template(

        "dashboard.html",

        total_prediction=total_prediction,

        stay_count=stay_count,

        leave_count=leave_count,

        latest_prediction=latest_prediction,

        metrics=metrics

    )


# ==========================================================
# HISTORY
# ==========================================================

@app.route("/history")
def history():

    histories = PredictionHistory.query.order_by(

        PredictionHistory.created_at.desc()

    ).all()

    return render_template(

        "history.html",

        histories=histories

    )


# ==========================================================
# HISTORY DETAIL
# ==========================================================

@app.route("/history/<int:history_id>")
def history_detail(history_id):

    history = PredictionHistory.query.get_or_404(

        history_id

    )

    return render_template(

        "history_detail.html",

        history=history

    )


# ==========================================================
# DELETE HISTORY
# ==========================================================

@app.route(

    "/history/delete/<int:history_id>",

    methods=["POST"]

)

def delete_history(history_id):

    history = PredictionHistory.query.get_or_404(

        history_id

    )

    db.session.delete(

        history

    )

    db.session.commit()

    flash(

        "History berhasil dihapus.",

        "success"

    )

    return redirect(

        url_for(

            "history"

        )

    )


# ==========================================================
# CLEAR HISTORY
# ==========================================================

@app.route(

    "/history/clear",

    methods=["POST"]

)

def clear_history():

    PredictionHistory.query.delete()

    db.session.commit()

    flash(

        "Seluruh history berhasil dihapus.",

        "success"

    )

    return redirect(

        url_for(

            "history"

        )

    )


# ==========================================================
# ABOUT
# ==========================================================

@app.route("/about")
def about():

    return render_template(

        "about.html",

        project_name=PROJECT_NAME,

        version=PROJECT_VERSION,

        author=AUTHOR,

        university=UNIVERSITY,

        study_program=STUDY_PROGRAM,

        framework=FRAMEWORK,

        ai_model=AI_MODEL,

        database=DATABASE

    )


# ==========================================================
# API STATISTICS
# ==========================================================

@app.route("/api/statistics")
def api_statistics():

    total_prediction = PredictionHistory.query.count()

    stay_count = PredictionHistory.query.filter_by(
        prediction="Stay"
    ).count()

    leave_count = PredictionHistory.query.filter_by(
        prediction="Leave"
    ).count()

    if total_prediction == 0:

        stay_percentage = 0

        leave_percentage = 0

    else:

        stay_percentage = round(

            stay_count / total_prediction * 100,

            2

        )

        leave_percentage = round(

            leave_count / total_prediction * 100,

            2

        )

    return jsonify({

        "total_prediction": total_prediction,

        "stay": stay_count,

        "leave": leave_count,

        "stay_percentage": stay_percentage,

        "leave_percentage": leave_percentage

    })

# ==========================================================
# CONTEXT PROCESSOR
# ==========================================================

@app.context_processor
def inject_global_variables():

    return {

        "project_name": PROJECT_NAME,

        "project_version": PROJECT_VERSION,

        "author": AUTHOR

    }


# ==========================================================
# ERROR 404
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):

    return (

        render_template(

            "404.html"

        ),

        404

    )


# ==========================================================
# ERROR 500
# ==========================================================

@app.errorhandler(500)
def internal_server_error(error):

    db.session.rollback()

    logging.exception(error)

    return (

        render_template(

            "500.html"

        ),

        500

    )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "application": PROJECT_NAME,

        "version": PROJECT_VERSION,

        "model_loaded": model is not None,

        "preprocessor_loaded": preprocessor is not None

    })


# ==========================================================
# APPLICATION INFO
# ==========================================================

@app.route("/info")
def info():

    return jsonify({

        "project": PROJECT_NAME,

        "version": PROJECT_VERSION,

        "author": AUTHOR,

        "university": UNIVERSITY,

        "study_program": STUDY_PROGRAM,

        "framework": FRAMEWORK,

        "ai_model": AI_MODEL,

        "database": DATABASE

    })


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print("EMPLOYEE ATTRITION PREDICTION")

    print("=" * 70)

    print()

    print(f"Project      : {PROJECT_NAME}")

    print(f"Version      : {PROJECT_VERSION}")

    print(f"Author       : {AUTHOR}")

    print(f"Framework    : {FRAMEWORK}")

    print(f"AI Model     : {AI_MODEL}")

    print(f"Database     : {DATABASE}")

    print()

    print(f"Running on http://127.0.0.1:{PORT}")

    print()

    print("=" * 70)

    print()

    try:

        app.run(

            host=HOST,

            port=PORT,

            debug=DEBUG

        )

    except Exception as error:

        logging.exception(error)

        print()

        print("=" * 70)

        print("APPLICATION FAILED")

        print("=" * 70)

        print()

        print(error)