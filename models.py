from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

Severity = Literal["immediate", "delayed", "minor"]
GRID_KM = 20 


class Location(BaseModel):
    x: float = Field(ge=0, le=GRID_KM)
    y: float = Field(ge=0, le=GRID_KM)


class Victim(BaseModel):
    victim_id: str
    severity: Severity
    deteriorates_at_min: Optional[int] = None  


class Incident(BaseModel):
    incident_id: str
    arrival_min: int = Field(ge=0)
    location: Location
    incident_type: str
    needs_responders: list[str]
    victims: list[Victim] = Field(min_length=1)
    priority_hint: int = Field(ge=1, le=3)  
    report: str = ""                         


class Hospital(BaseModel):
    hospital_id: str
    name: str
    kind: str
    location: Location
    er_beds: int = Field(gt=0)
    beds_available: int = Field(ge=0)
    specialties: list[str]

    @model_validator(mode="after")
    def beds_ok(self):
        if self.beds_available > self.er_beds:
            raise ValueError(f"{self.hospital_id}: more free beds than ER beds")
        return self


class Staff(BaseModel):
    staff_id: str
    role: Literal["doctor", "nurse"]
    hospital_id: str
    skills: list[str] = Field(min_length=1)
    workload_cap: int = Field(gt=0)
    current_load: int = 0
    on_shift: bool = True


class Ambulance(BaseModel):
    ambulance_id: str
    level: Literal["ALS", "BLS"]         
    location: Location
    capacity: int = Field(gt=0, le=2)
    status: Literal["available", "en_route", "transporting", "busy"] = "available"


class ResponderTeam(BaseModel):
    team_id: str
    team_type: Literal["fire", "rescue", "hazmat"]
    location: Location
    status: Literal["available", "on_scene", "busy"] = "available"


class World(BaseModel):
    version: str
    seed: int
    grid_km: float
    horizon_min: int
    ambulance_speed_kmh: float
    traffic_factor: tuple[float, float]
    hospitals: list[Hospital]
    staff: list[Staff]
    ambulances: list[Ambulance]
    responders: list[ResponderTeam]
    incidents: list[Incident]

    @model_validator(mode="after")
    def cross_checks(self):
        hospital_ids = {h.hospital_id for h in self.hospitals}
        for s in self.staff:
            if s.hospital_id not in hospital_ids:
                raise ValueError(f"{s.staff_id} works at unknown hospital {s.hospital_id}")
        ids = [i.incident_id for i in self.incidents]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate incident ids")
        if any(i.arrival_min > self.horizon_min for i in self.incidents):
            raise ValueError("incident arrives after the simulation ends")
        return self
