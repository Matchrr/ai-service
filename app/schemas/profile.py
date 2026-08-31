from pydantic import BaseModel, Field


class ExperiencePayload(BaseModel):
    title: str = ""
    company: str = ""
    start_date: str | None = None
    end_date: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ProfilePayload(BaseModel):
    full_name: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperiencePayload] = Field(default_factory=list)
    volunteering: list[ExperiencePayload] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
