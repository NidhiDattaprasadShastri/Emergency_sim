from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Lenient(BaseModel):
    @field_validator("reason", mode="before", check_fields=False)
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else str(v)


class Assess(Lenient):
    incident_id: str
    priority: int = Field(ge=1, le=3)     
    immediate: int = Field(ge=0)
    delayed: int = Field(ge=0)
    minor: int = Field(ge=0)
    reason: str = ""


class AssessmentOut(BaseModel):
    assessments: list[Assess]


class AmbProposal(Lenient):
    ambulance_id: str
    incident_id: str
    reason: str = ""


class TeamProposal(Lenient):
    team_id: str
    incident_id: str
    reason: str = ""


class DispatchOut(BaseModel):
    ambulance_proposals: list[AmbProposal] = []
    team_proposals: list[TeamProposal] = []


class Allocation(Lenient):
    ambulance_id: str
    incident_id: str
    victim_count: int = Field(ge=1, le=2)
    hospital_id: str
    reason: str = ""


class AllocationOut(BaseModel):
    decisions: list[Allocation]


class HospitalResponse(Lenient):
    request_id: str
    decision: Literal["ACCEPT", "REJECT", "COUNTER"]
    alt_hospital_id: Optional[str] = None
    reason: str = ""

    @field_validator("decision", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).strip().upper()


class HospitalOut(BaseModel):
    responses: list[HospitalResponse]


class StaffDecision(Lenient):
    decision: Literal["OFFER", "DECLINE"]
    reason: str = ""

    @field_validator("decision", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).strip().upper()