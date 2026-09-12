from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """Request body for POST /predict."""

    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Congratulations! You have won $1,000"
            }
        }
    )


class Prediction(BaseModel):
    """Response body for POST /predict."""

    message: str
    prediction: str