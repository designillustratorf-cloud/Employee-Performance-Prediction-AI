"""
=========================================================
database.py
Employee Attrition Prediction Using ANN
=========================================================
"""

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionHistory(db.Model):
    """
    Menyimpan seluruh riwayat prediksi
    """

    __tablename__ = "prediction_history"

    # =====================================================
    # PRIMARY KEY
    # =====================================================

    id = db.Column(
        db.Integer,
        primary_key=True,
        autoincrement=True
    )

    # =====================================================
    # EMPLOYEE INFORMATION
    # =====================================================

    age = db.Column(
        db.Integer,
        nullable=False
    )

    gender = db.Column(
        db.String(30),
        nullable=False
    )

    marital_status = db.Column(
        db.String(30),
        nullable=False
    )

    department = db.Column(
        db.String(100),
        nullable=False
    )

    job_role = db.Column(
        db.String(100),
        nullable=False
    )

    job_level = db.Column(
        db.Integer,
        nullable=False
    )

    monthly_income = db.Column(
        db.Float,
        nullable=False
    )

    hourly_rate = db.Column(
        db.Float,
        nullable=False
    )

    years_at_company = db.Column(
        db.Integer,
        nullable=False
    )

    years_in_current_role = db.Column(
        db.Integer,
        nullable=False
    )

    years_since_last_promotion = db.Column(
        db.Integer,
        nullable=False
    )

    work_life_balance = db.Column(
        db.Integer,
        nullable=False
    )

    job_satisfaction = db.Column(
        db.Integer,
        nullable=False
    )

    performance_rating = db.Column(
        db.Integer,
        nullable=False
    )

    training_hours_last_year = db.Column(
        db.Integer,
        nullable=False
    )

    overtime = db.Column(
        db.String(10),
        nullable=False
    )

    project_count = db.Column(
        db.Integer,
        nullable=False
    )

    average_hours_worked_per_week = db.Column(
        db.Float,
        nullable=False
    )

    absenteeism = db.Column(
        db.Float,
        nullable=False
    )

    work_environment_satisfaction = db.Column(
        db.Integer,
        nullable=False
    )

    relationship_with_manager = db.Column(
        db.Integer,
        nullable=False
    )

    job_involvement = db.Column(
        db.Integer,
        nullable=False
    )

    distance_from_home = db.Column(
        db.Float,
        nullable=False
    )

    number_of_companies_worked = db.Column(
        db.Integer,
        nullable=False
    )

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    prediction = db.Column(
        db.String(10),
        nullable=False
    )

    probability = db.Column(
        db.Float,
        nullable=False
    )

    # =====================================================
    # TIMESTAMP
    # =====================================================

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # =====================================================
    # SERIALIZE
    # =====================================================

    def to_dict(self):

        return {

            "id": self.id,

            "age": self.age,

            "gender": self.gender,

            "marital_status": self.marital_status,

            "department": self.department,

            "job_role": self.job_role,

            "job_level": self.job_level,

            "monthly_income": self.monthly_income,

            "hourly_rate": self.hourly_rate,

            "years_at_company": self.years_at_company,

            "years_in_current_role":
                self.years_in_current_role,

            "years_since_last_promotion":
                self.years_since_last_promotion,

            "work_life_balance":
                self.work_life_balance,

            "job_satisfaction":
                self.job_satisfaction,

            "performance_rating":
                self.performance_rating,

            "training_hours_last_year":
                self.training_hours_last_year,

            "overtime":
                self.overtime,

            "project_count":
                self.project_count,

            "average_hours_worked_per_week":
                self.average_hours_worked_per_week,

            "absenteeism":
                self.absenteeism,

            "work_environment_satisfaction":
                self.work_environment_satisfaction,

            "relationship_with_manager":
                self.relationship_with_manager,

            "job_involvement":
                self.job_involvement,

            "distance_from_home":
                self.distance_from_home,

            "number_of_companies_worked":
                self.number_of_companies_worked,

            "prediction":
                self.prediction,

            "probability":
                self.probability,

            "created_at":
                self.created_at.strftime(
                    "%d-%m-%Y %H:%M:%S"
                )

        }

    # =====================================================
    # STRING REPRESENTATION
    # =====================================================

    def __repr__(self):

        return (
            f"<PredictionHistory "
            f"id={self.id}, "
            f"prediction={self.prediction}, "
            f"probability={self.probability:.2f}%>"
        )


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db(app):

    db.init_app(app)

    with app.app_context():

        db.create_all()

        print("=" * 60)
        print("DATABASE INITIALIZED")
        print("=" * 60)