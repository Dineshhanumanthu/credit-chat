from pathlib import Path
from datetime import datetime, timezone
import uuid
import json

import numpy as np
import pandas as pd


class ModelService:

    def __init__(self, base: Path):
        self.base = Path(base)

        # Your actual project structure:
        # D:\credit-chat\credit_risk_model.pkl
        self.model_file = self.base / "credit_risk_model.pkl"

        # Dataset files
        self.dataset_file = self.base / "data" / "credit_risk_dataset.csv"
        self.data_file = self.base / "data" / "credit_risk_default.csv"

        self.model = None
        self.model_error = None

        self._load_model()
        self._ensure_store()

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    def _load_model(self):
        try:
            import joblib

            self.model = joblib.load(self.model_file)

        except Exception as exc:
            self.model = None
            self.model_error = str(exc)

    # ---------------------------------------------------------
    # CSV STORAGE
    # ---------------------------------------------------------

    def _ensure_store(self):

        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
            "customer_id",
            "customer_username",
            "recommended_loan",
            "recommended_products",
            "staff_decision",
            "accepted_product",
            "submission_date",
            "application_id",
            "submitted_at",
            "loan_tenure",
            "existing_emi",
            "emi_to_income_ratio",
            "default_probability",
            "risk_level",
            "status",
        ]

        if not self.data_file.exists():

            pd.DataFrame(columns=columns).to_csv(
                self.data_file,
                index=False
            )

    def _read_applications(self):

        self._ensure_store()

        try:
            return pd.read_csv(
                self.data_file,
                on_bad_lines="skip"
            ).fillna("")

        except (pd.errors.EmptyDataError, pd.errors.ParserError):

            # Re-create a clean application file
            self.data_file.unlink(missing_ok=True)
            self._ensure_store()

            return pd.read_csv(
                self.data_file
            ).fillna("")

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    def _features(self, p):

        x = dict(p)

        income = float(x["person_income"])
        loan = float(x["loan_amnt"])
        age = int(x["person_age"])

        # Ratios
        x["income_to_loan"] = (
            income / loan
            if loan
            else 0.0
        )

        x["loan_to_income"] = (
            loan / income
            if income
            else 0.0
        )

        # Age group
        if age < 25:
            x["age_group"] = "Young"

        elif age < 35:
            x["age_group"] = "Adult"

        elif age < 50:
            x["age_group"] = "Middle-aged"

        else:
            x["age_group"] = "Senior"

        # Income group
        if income < 30000:
            x["income_group"] = "Low"

        elif income < 60000:
            x["income_group"] = "Medium"

        elif income < 100000:
            x["income_group"] = "High"

        else:
            x["income_group"] = "Very High"

        return pd.DataFrame([x])

    # ---------------------------------------------------------
    # PRODUCT RECOMMENDATION
    # ---------------------------------------------------------

    def _recommend_products(self, p):

        products = []

        intent = str(
            p["loan_intent"]
        ).lower()

        grade = str(
            p["loan_grade"]
        ).upper()

        income = float(
            p["person_income"]
        )

        # Loan-intent recommendations
        if "education" in intent:

            products.extend([
                "Education Loan",
                "Savings Account"
            ])

        elif "home" in intent:

            products.extend([
                "Home Loan",
                "Home Insurance"
            ])

        elif "medical" in intent:

            products.extend([
                "Health Insurance",
                "Personal Loan"
            ])

        elif "debt" in intent:

            products.extend([
                "Balance Transfer",
                "Credit Card"
            ])

        elif "venture" in intent:

            products.extend([
                "Business Account",
                "Business Credit"
            ])

        else:

            products.extend([
                "Personal Loan",
                "Credit Card"
            ])

        # Income-based recommendation
        if income >= 60000:

            products.append(
                "Premium Savings"
            )

        else:

            products.append(
                "Savings Account"
            )

        # Grade-based recommendation
        if grade in {"A", "B"}:

            products.append(
                "Rewards Credit Card"
            )

        # Remove duplicates while preserving order
        products = list(
            dict.fromkeys(products)
        )

        return products[:4]

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    def predict_and_recommend(self, p):

        X = self._features(p)

        probability = None
        risk = "Unavailable"

        if self.model is None:

            raise RuntimeError(
                "The ML model could not be loaded.\n\n"
                f"Model path: {self.model_file}\n\n"
                f"Error: {self.model_error}"
            )

        try:

            probability = float(
                self.model.predict_proba(X)[0, 1]
            )

        except Exception as exc:

            raise RuntimeError(
                "The model could not score this application.\n\n"
                "Make sure scikit-learn==1.6.1 is installed "
                "because the supplied model was trained with "
                "that version.\n\n"
                f"Details: {exc}"
            )

        # Risk classification
        if probability < 0.20:

            risk = "Low"

        elif probability < 0.50:

            risk = "Medium"

        else:

            risk = "High"

        products = self._recommend_products(p)

        return {
            "default_probability": probability,
            "risk_level": risk,
            "recommended_loan": str(
                p["loan_intent"]
            ).replace(
                "_",
                " "
            ).title(),
            "recommended_products": products,
        }

    # ---------------------------------------------------------
    # SAVE APPLICATION
    # ---------------------------------------------------------

    def save_application(self, p):

        result = self.predict_and_recommend(p)

        now = datetime.now(
            timezone.utc
        ).isoformat()

        df = self._read_applications()

        application_id = (
            "APP-"
            + uuid.uuid4().hex[:8].upper()
        )

        # Generate customer ID safely
        if (
            "customer_id" in df.columns
            and len(df) > 0
        ):

            ids = pd.to_numeric(
                df["customer_id"],
                errors="coerce"
            ).dropna()

            customer_id = (
                int(ids.max()) + 1
                if len(ids)
                else 1
            )

        else:

            customer_id = 1

        income = float(
            p["person_income"]
        )

        existing_emi = float(
            p.get(
                "existing_emi",
                0
            )
        )

        emi_ratio = (
            existing_emi / income
            if income
            else 0
        )

        row = {

            "person_age":
                p.get("person_age"),

            "person_income":
                p.get("person_income"),

            "person_home_ownership":
                p.get("person_home_ownership"),

            "person_emp_length":
                p.get("person_emp_length"),

            "loan_intent":
                p.get("loan_intent"),

            "loan_grade":
                p.get("loan_grade"),

            "loan_amnt":
                p.get("loan_amnt"),

            "loan_int_rate":
                p.get("loan_int_rate"),

            "loan_percent_income":
                p.get("loan_percent_income"),

            "cb_person_default_on_file":
                p.get("cb_person_default_on_file"),

            "cb_person_cred_hist_length":
                p.get("cb_person_cred_hist_length"),

            "customer_id":
                customer_id,

            "customer_username":
                p.get("customer_username"),

            "recommended_loan":
                result["recommended_loan"],

            "recommended_products":
                json.dumps(
                    result["recommended_products"]
                ),

            "staff_decision":
                "",

            "accepted_product":
                "",

            "submission_date":
                now[:10],

            "application_id":
                application_id,

            "submitted_at":
                now,

            "loan_tenure":
                p.get("loan_tenure", ""),

            "existing_emi":
                existing_emi,

            "emi_to_income_ratio":
                emi_ratio,

            "default_probability":
                result["default_probability"],

            "risk_level":
                result["risk_level"],

            "status":
                "Pending",
        }

        new_row = pd.DataFrame(
            [row]
        )

        out = pd.concat(
            [
                df,
                new_row
            ],
            ignore_index=True
        )

        out.to_csv(
            self.data_file,
            index=False
        )

        return {
            "application": row,
            "recommendation": result
        }

    # ---------------------------------------------------------
    # APPLICATIONS
    # ---------------------------------------------------------

    def list_applications(self):

        df = self._read_applications()

        return df.to_dict(
            orient="records"
        )

    def get_application(
        self,
        application_id
    ):

        df = self._read_applications()

        if "application_id" not in df.columns:
            return None

        mask = (
            df["application_id"]
            .astype(str)
            == str(application_id)
        )

        if not mask.any():
            return None

        return df.loc[
            mask
        ].iloc[0].to_dict()

    # ---------------------------------------------------------
    # STAFF DECISION
    # ---------------------------------------------------------

    def update_decision(
        self,
        application_id,
        decision,
        product
    ):

        df = self._read_applications()

        if "application_id" not in df.columns:
            return None

        mask = (
            df["application_id"]
            .astype(str)
            == str(application_id)
        )

        if not mask.any():
            return None

        idx = df.index[mask][0]

        decision = str(
            decision
        ).strip()

        if decision not in {
            "Approved",
            "Rejected"
        }:

            raise ValueError(
                "Decision must be Approved or Rejected."
            )

        df.loc[
            idx,
            "staff_decision"
        ] = decision

        df.loc[
            idx,
            "accepted_product"
        ] = product or ""

        df.loc[
            idx,
            "status"
        ] = decision

        df.to_csv(
            self.data_file,
            index=False
        )

        return df.loc[
            idx
        ].to_dict()

    # ---------------------------------------------------------
    # DATASET METRICS
    # ---------------------------------------------------------

    def metrics(self):

        if not self.dataset_file.exists():

            return {
                "rows": 0,
                "default_rate": 0,
                "avg_loan_amount": 0,
                "algorithm":
                    "RandomForestClassifier",
                "trees": 200,
                "max_depth": 8,
            }

        df = pd.read_csv(
            self.dataset_file
        )

        result = {
            "rows": int(len(df)),
            "default_rate": float(
                df["loan_status"]
                .astype(int)
                .mean()
            ),
            "avg_loan_amount": float(
                df["loan_amnt"].mean()
            ),
            "algorithm":
                "RandomForestClassifier",
            "trees": 200,
            "max_depth": 8,
        }

        return result
