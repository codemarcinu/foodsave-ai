from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional

import pytz  # Needed for timezone validation
from pydantic import BaseModel, Field, validator
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from ..core.database import Base

# Removed unused import


class InteractionType(str, Enum):
    """Types of user interactions for analysis"""

    QUERY = "query"
    CLICK = "click"
    FEEDBACK = "feedback"
    SETTING_CHANGE = "setting_change"


class Formality(str, Enum):
    """Communication style formality options"""

    FORMAL = "formal"
    NEUTRAL = "neutral"
    CASUAL = "casual"


class UserPreferences(BaseModel):
    """User preferences for personalization"""

    formality: Formality = Formality.NEUTRAL
    news_topics: List[str] = []
    favorite_locations: List[str] = []
    notifications_enabled: bool = True
    daily_summary_enabled: bool = False
    alert_severe_weather: bool = True
    time_format_24h: bool = True
    temperature_unit: str = "celsius"


class UserSchedule(BaseModel):
    """User schedule preferences for automated suggestions"""

    wake_time: time = Field(default_factory=lambda: time(7, 0))
    sleep_time: time = Field(default_factory=lambda: time(23, 0))
    work_days: List[int] = [0, 1, 2, 3, 4]  # Monday=0, Sunday=6
    work_start_time: time = Field(default_factory=lambda: time(9, 0))
    work_end_time: time = Field(default_factory=lambda: time(17, 0))
    lunch_time: time = Field(default_factory=lambda: time(12, 0))
    time_zone: str = "Europe/Warsaw"

    @validator("time_zone")
    def validate_timezone(cls, v):
        if v not in pytz.all_timezones:
            return "Europe/Warsaw"  # Default to Warsaw if invalid
        return v


class UserActivity(Base):
    """Database model for tracking user activity"""

    __tablename__ = "user_activities"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_profiles.user_id"), nullable=False)
    interaction_type = Column(String, nullable=False)
    content = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.now)
    activity_metadata = Column(
        JSON, nullable=True
    )  # Zmieniono z 'metadata' na 'activity_metadata'

    user = relationship("UserProfile", back_populates="activities")


class UserProfile(Base):
    """Database model for user profiles"""

    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.now)
    last_active = Column(
        DateTime(timezone=True), default=datetime.now, onupdate=datetime.now
    )
    preferences = Column(JSON, nullable=False, default=lambda: UserPreferences().dict())
    schedule = Column(JSON, nullable=False, default=lambda: UserSchedule().dict())
    topics_of_interest = Column(JSON, nullable=False, default=list)

    activities = relationship(
        "UserActivity", back_populates="user", cascade="all, delete-orphan"
    )

    def get_preferences(self) -> UserPreferences:
        """Get user preferences as pydantic model"""
        return UserPreferences.parse_obj(self.preferences)

    def get_schedule(self) -> UserSchedule:
        """Get user schedule as pydantic model"""
        return UserSchedule.parse_obj(self.schedule)


class UserProfileData(BaseModel):
    """Pydantic model for user profile data"""

    user_id: str
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    schedule: UserSchedule = Field(default_factory=UserSchedule)
    topics_of_interest: List[str] = []
    activity_stats: Dict[str, int] = {}
    last_active: Optional[datetime] = None


# Create instance - will be initialized properly when database is available
profile_manager = None
