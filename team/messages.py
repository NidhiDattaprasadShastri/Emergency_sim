from typing import Literal, Optional
from pydantic import BaseModel

MsgType = Literal["PRIORITY", "PROPOSAL", "CONFLICT", "ADMISSION_REQUEST",
                  "ACCEPT", "REJECT", "COUNTER", "DECISION", "UNRESOLVED",
                  "STAFF_REQUEST", "STAFF_OFFER", "STAFF_DECLINE"]

class Message(BaseModel):
    t: int                        
    sender: str
    recipient: str
    msg_type: MsgType
    incident_id: Optional[str] = None
    proposed_action: str            
    reason: str = ""
    decided_by: str = "rule"       