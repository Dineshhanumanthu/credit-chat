from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.model_service import ModelService

BASE = Path(__file__).resolve().parents[1]
DATA_FILE = BASE / "data" / "credit_risk_default.csv"

app = FastAPI(title="Bank Credit Risk & Cross-Sell Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService(BASE)

STAFF = {"staff": "staff123"}
DEVELOPER = {"developer": "dev123"}

class LoginRequest(BaseModel):
    username: str
    password: str
    role: str

class Application(BaseModel):
    customer_username: str = Field(min_length=1)
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(ge=0, le=80)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0)
    loan_percent_income: float = Field(ge=0)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(ge=0, le=100)
    loan_tenure: int = Field(default=36, ge=1, le=120)
    existing_emi: float = Field(default=0, ge=0)

class StaffDecision(BaseModel):
    staff_decision: str
    accepted_product: Optional[str] = None

def auth(role: str, authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    token = authorization.split(" ", 1)[1]
    expected = f"{role}:"
    if not token.startswith(expected):
        raise HTTPException(403, "Insufficient permissions")
    return token[len(expected):]

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/login")
def login(req: LoginRequest):
    if req.role == "customer":
        if not req.username or not req.password:
            raise HTTPException(401, "Username and password are required")
        return {"ok": True, "role": "customer", "token": f"customer:{req.username}", "username": req.username}
    creds = STAFF if req.role == "staff" else DEVELOPER if req.role == "developer" else {}
    if creds.get(req.username) != req.password:
        raise HTTPException(401, "Invalid credentials")
    return {"ok": True, "role": req.role, "token": f"{req.role}:{req.username}", "username": req.username}

@app.post("/api/applications/preview")
def preview(app: Application, authorization: Optional[str] = Header(None)):
    auth("customer", authorization)
    return service.predict_and_recommend(app.model_dump())

@app.post("/api/applications")
def create_application(app: Application, authorization: Optional[str] = Header(None)):
    username = auth("customer", authorization)
    payload = app.model_dump()
    payload["customer_username"] = username
    return service.save_application(payload)

@app.get("/api/applications")
def applications(authorization: Optional[str] = Header(None)):
    auth("staff", authorization)
    return service.list_applications()

@app.get("/api/applications/{application_id}")
def application(application_id: str, authorization: Optional[str] = Header(None)):
    auth("staff", authorization)
    item = service.get_application(application_id)
    if item is None:
        raise HTTPException(404, "Application not found")
    return item

@app.patch("/api/applications/{application_id}")
def decision(application_id: str, decision: StaffDecision, authorization: Optional[str] = Header(None)):
    auth("staff", authorization)
    if decision.staff_decision not in {"Approved", "Rejected"}:
        raise HTTPException(400, "Decision must be Approved or Rejected")
    item = service.update_decision(application_id, decision.staff_decision, decision.accepted_product)
    if item is None:
        raise HTTPException(404, "Application not found")
    return item

@app.get("/api/developer/metrics")
def metrics(authorization: Optional[str] = Header(None)):
    auth("developer", authorization)
    return service.metrics()
