from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from backend.agents.base_agent import AgentResponse, BaseAgent
from backend.agents.tools.tools import get_available_products_from_pantry
from backend.core.llm_client import llm_client


class ChefAgentInput(BaseModel):
    """Input model for ChefAgent"""

    ingredients: List[str] = Field(
        ..., min_items=1, description="List of available ingredients"
    )
    dietary_restrictions: Optional[str] = Field(
        None, description="Dietary restrictions"
    )
    model: Optional[str] = Field(
        "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0", description="LLM model to use"
    )


class RecipeSuggestion(BaseModel):
    """Model for recipe suggestion response"""

    recipe: str
    used_ingredients: List[Dict[str, Any]]


class ChefAgent(BaseAgent):
    """Agent that suggests recipes based on available pantry items"""

    def __init__(self, name: str = "ChefAgent"):
        super().__init__(name)

    async def process(self, input_data: Any) -> AgentResponse:
        """Main processing method - validates input and generates recipe"""
        try:
            # Validate input
            validated_input = ChefAgentInput.parse_obj(input_data)

            # Generate recipe
            return await self._generate_recipe(
                ingredients=validated_input.ingredients,
                dietary_restrictions=validated_input.dietary_restrictions,
                model=validated_input.model,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=str(e),
                text=f"Przepraszam, wystąpił błąd: {str(e)}",
            )

    async def _generate_recipe(
        self,
        ingredients: List[str],
        dietary_restrictions: Optional[str] = None,
        model: str = "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0",
    ) -> AgentResponse:
        """Generate recipe from given ingredients"""
        if not ingredients:
            return AgentResponse(
                success=False,
                error="No ingredients provided",
                text="Proszę podać składniki",
            )

        # Prepare prompt
        prompt = (
            "Mam następujące składniki:\n"
            f"{', '.join(ingredients)}\n\n"
            "Proszę zaproponuj prosty przepis wykorzystujący te składniki."
        )

        if dietary_restrictions:
            prompt += f"\n\nUwzględnij następujące ograniczenia dietetyczne: {dietary_restrictions}"

        async def recipe_generator():
            try:
                # Call LLM with streaming
                response = await llm_client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Jesteś pomocnym szefem kuchni."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                )

                # Stream the response chunks
                full_response = ""
                async for chunk in response:
                    content = chunk["message"]["content"]
                    full_response += content
                    yield content

            except Exception as e:
                yield f"Przepraszam, wystąpił błąd: {str(e)}"

        return AgentResponse(
            success=True,
            text_stream=recipe_generator(),
            message="Recipe generation started",
        )

    async def generate_recipe_idea(
        self, db: Any, model: str = "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"
    ) -> AgentResponse:
        """
        Generates recipe ideas based on available pantry items.

        Args:
            db: Database session
            model: LLM model to use for generating the recipe (default: llama3)

        Returns:
            AgentResponse with recipe suggestion or error message
        """
        # Get available products from pantry
        products = await get_available_products_from_pantry(db)

        if not products:
            return AgentResponse(
                success=True,
                text="Twoja spiżarnia jest pusta!",
                message="Pantry is empty",
            )

        # Prepare list of available products for the prompt
        product_list = "\n".join(
            f"- {product.name} (ID: {product.id})" for product in products
        )

        # Create LLM prompt
        prompt = (
            "Mam następujące produkty w spiżarni:\n"
            f"{product_list}\n\n"
            "Proszę zaproponuj prosty przepis wykorzystujący te produkty. "
            "Odpowiedz w formacie:\n"
            "PRZEPIS: [treść przepisu]\n"
            "UŻYTE SKŁADNIKI: [lista nazw użytych składników]"
        )

        async def response_generator():
            # Call LLM with specified model and stream the response
            full_response = ""
            async for chunk in llm_client.generate_stream(
                model=model,
                messages=[
                    {"role": "system", "content": "Jesteś pomocnym szefem kuchni."},
                    {"role": "user", "content": prompt},
                ],
            ):
                content = chunk["message"]["content"]
                full_response += content
                yield content

            # After streaming, parse the full response to extract used ingredients
            used_ingredients = []
            if "UŻYTE SKŁADNIKI:" in full_response:
                parts = full_response.split("UŻYTE SKŁADNIKI:")
                ingredient_names = [
                    name.strip() for name in parts[1].split(",") if name.strip()
                ]
                used_ingredients = [
                    {"id": p.id, "name": p.name}
                    for p in products
                    if p.name in ingredient_names
                ]
            print(f"Used ingredients identified: {used_ingredients}")

        return AgentResponse(
            success=True,
            text_stream=response_generator(),
            message="Recipe stream started.",
        )
