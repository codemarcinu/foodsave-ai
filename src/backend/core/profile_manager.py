from datetime import datetime, time
from typing import Dict, List, Optional, Union

import pytz

from ..models.user_profile import (
    InteractionType,
    UserPreferences,
    UserProfileData,
    UserSchedule,
)


class ProfileManager:
    """Manager for user profiles with personalization logic"""

    def __init__(self, db_session):
        self.db = db_session
        self.active_sessions: Dict[str, UserProfileData] = {}

    async def get_or_create_profile(self, session_id: str) -> UserProfileData:
        """Get or create a user profile for session"""
        # Check if we have it in memory
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Look up in database
        from src.backend.core.crud import (
            create_user_profile,
            get_user_profile_by_session,
        )

        profile = await get_user_profile_by_session(self.db, session_id)
        if not profile:
            # Create a new profile
            user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            profile = await create_user_profile(self.db, user_id, session_id)

        # Convert to UserProfileData
        profile_data = UserProfileData(
            user_id=profile.user_id,
            preferences=UserPreferences.parse_obj(profile.preferences),
            schedule=UserSchedule.parse_obj(profile.schedule),
            topics_of_interest=profile.topics_of_interest,
            last_active=profile.last_active,
        )

        # Cache in memory
        self.active_sessions[session_id] = profile_data

        return profile_data

    async def update_preferences(
        self, session_id: str, preferences: Union[UserPreferences, Dict]
    ) -> UserProfileData:
        """Update user preferences"""
        profile_data = await self.get_or_create_profile(session_id)

        # Convert dict to model if needed
        if isinstance(preferences, dict):
            preferences = UserPreferences(**preferences)

        # Update in memory
        profile_data.preferences = preferences

        # Update in database
        from src.backend.core.crud import update_user_preferences

        await update_user_preferences(self.db, profile_data.user_id, preferences.dict())

        return profile_data

    async def update_schedule(
        self, session_id: str, schedule: Union[UserSchedule, Dict]
    ) -> UserProfileData:
        """Update user schedule"""
        profile_data = await self.get_or_create_profile(session_id)

        # Convert dict to model if needed
        if isinstance(schedule, dict):
            schedule = UserSchedule(**schedule)

        # Update in memory
        profile_data.schedule = schedule

        # Update in database
        from src.backend.core.crud import update_user_schedule

        await update_user_schedule(self.db, profile_data.user_id, schedule.dict())

        return profile_data

    async def log_activity(
        self,
        session_id: str,
        interaction_type: InteractionType,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log user activity for analysis"""
        profile_data = await self.get_or_create_profile(session_id)

        # Create activity record
        from src.backend.core.user_activity import create_user_activity

        await create_user_activity(
            self.db, profile_data.user_id, interaction_type, content, metadata
        )

        # Update activity stats in memory
        if interaction_type.value not in profile_data.activity_stats:
            profile_data.activity_stats[interaction_type.value] = 0
        profile_data.activity_stats[interaction_type.value] += 1

    async def get_personalized_suggestions(
        self, session_id: str, current_time: Optional[datetime] = None
    ) -> List[str]:
        """Get time and schedule based suggestions"""
        profile_data = await self.get_or_create_profile(session_id)
        schedule = profile_data.schedule

        if not current_time:
            current_time = datetime.now(pytz.timezone(schedule.time_zone))

        current_time_only = current_time.time()
        weekday = current_time.weekday()

        suggestions = []

        # Morning routine suggestions
        if (
            schedule.wake_time
            <= current_time_only
            <= time(hour=schedule.wake_time.hour + 2, minute=0)
        ):
            suggestions.append("Dzień dobry! Jak mogę pomóc z poranną rutyną?")
            suggestions.append("Czy chcesz zobaczyć prognozę pogody na dziś?")

        # Work time suggestions
        is_work_day = weekday in schedule.work_days
        if (
            is_work_day
            and schedule.work_start_time <= current_time_only <= schedule.work_end_time
        ):
            # Lunch time suggestion
            lunch_start = time(
                hour=schedule.lunch_time.hour - 1, minute=schedule.lunch_time.minute
            )
            lunch_end = time(
                hour=schedule.lunch_time.hour + 1, minute=schedule.lunch_time.minute
            )

            if lunch_start <= current_time_only <= lunch_end:
                suggestions.append(
                    "Czas na przerwę na lunch! Potrzebujesz przepisu na szybki posiłek?"
                )

            # Work break reminder
            if (
                len(suggestions) == 0
                and current_time_only.hour % 2 == 0
                and current_time_only.minute < 15
            ):
                suggestions.append("Pamiętaj o krótkiej przerwie dla zdrowia!")

        # Evening routine
        evening_start = time(hour=20, minute=0)
        if evening_start <= current_time_only <= schedule.sleep_time:
            suggestions.append("Czy potrzebujesz planu na jutro?")

        return suggestions

    async def analyze_interests(self, session_id: str) -> List[str]:
        """Analyze user activities to determine interests"""
        from src.backend.core.crud import get_user_activities

        profile_data = await self.get_or_create_profile(session_id)

        # Get recent activities
        activities = await get_user_activities(self.db, profile_data.user_id, limit=50)

        # Simple keyword extraction from queries
        interest_counters: Dict[str, int] = {}

        for activity in activities:
            if (
                activity.interaction_type == InteractionType.QUERY.value
                and activity.content
            ):
                # Extract possible interests from queries
                words = activity.content.lower().split()
                for word in words:
                    if len(word) > 3 and word not in [
                        "czym",
                        "jaki",
                        "jaka",
                        "jakie",
                        "gdzie",
                        "kiedy",
                    ]:
                        if word in interest_counters:
                            interest_counters[word] += 1
                        else:
                            interest_counters[word] = 1

        # Find top interests
        sorted_interests = sorted(
            interest_counters.items(), key=lambda x: x[1], reverse=True
        )
        top_interests = [word for word, count in sorted_interests[:10] if count > 1]

        # Update topics of interest
        if top_interests:
            profile_data.topics_of_interest = top_interests

            # Update in database
            from src.backend.core.crud import update_user_topics

            await update_user_topics(self.db, profile_data.user_id, top_interests)

        return profile_data.topics_of_interest
