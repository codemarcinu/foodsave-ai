from abc import ABC, abstractmethod
from typing import Any, Dict


class AgentPlugin(ABC):
    """Base interface for agent plugins"""

    @abstractmethod
    def initialize(self, agent: Any) -> None:
        """Initialize plugin with agent instance"""
        pass

    @abstractmethod
    def before_process(self, input_data: Dict) -> Dict:
        """Pre-process input data"""
        pass

    @abstractmethod
    def after_process(self, output_data: Dict) -> Dict:
        """Post-process output data"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        pass
