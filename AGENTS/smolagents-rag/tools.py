from smolagents import DuckDuckGoSearchTool
from smolagents import Tool
import random
from smolagents import Tool
from huggingface_hub import list_models
from smolagents import CodeAgent, HfApiModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()

# Example usage
results = search_tool("Who's the current President of France?")
print(results)


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        # Randomly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = WeatherInfoTool()


class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
            
            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = HubStatsTool()

# Example usage
print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook


# Initialize the Hugging Face model
model = HfApiModel(token=os.getenv('HUGGINGFACE_TOKEN'))

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[search_tool, weather_info_tool, hub_stats_tool], 
    model=model
)

# Example query Alfred might receive during the gala
response = alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)

