from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import base64
import os
import io
import json
import logging
import requests
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Updated endpoint to include API key in URL
def get_gemini_endpoint():
    return f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def encode_image_to_base64(image_data):
    try:
        # Convert bytes to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise

def get_recipe_from_image(image_data):
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_data)
        
        # Prepare the prompt
        prompt = """
        Look at this food image and:
        1. Identify the dish.
        2. Generate a detailed recipe for it including:
           - List of ingredients with measurements
           - Step-by-step cooking instructions
           - Cooking time and servings
        Format the response as JSON with the following structure:
        {
            "name": "Name of the dish",
            "description": "Brief description",
            "ingredients": ["ingredient 1", "ingredient 2", ...],
            "instructions": ["step 1", "step 2", ...],
            "prepTime": "preparation time",
            "cookTime": "cooking time",
            "servings": "number of servings"
        }
        """

        # Prepare the request
        headers = {
            'Content-Type': 'application/json'
        }

        data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }]
        }

        # Make request to Gemini API
        endpoint = get_gemini_endpoint()
        logger.debug(f"Making request to Gemini API at endpoint: {endpoint}")
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()

        # Parse response
        response_data = response.json()
        logger.debug(f"Gemini API Response: {response_data}")

        if 'candidates' in response_data and response_data['candidates']:
            recipe_text = response_data['candidates'][0]['content']['parts'][0]['text']
            recipe_text = recipe_text.replace('```json', '').replace('```', '').strip()
            recipe_data = json.loads(recipe_text)
            return recipe_data
        else:
            raise ValueError("No valid response from Gemini API")

    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        raise

    
def generate_html_from_recipe(recipe_data):
    try:
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{recipe_data['name']} Recipe</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .description {{
                    font-style: italic;
                    color: #666;
                    margin-bottom: 20px;
                }}
                .recipe-info {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                    color: #666;
                }}
                .ingredients, .instructions {{
                    margin-bottom: 30px;
                }}
                ul, ol {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>{recipe_data['name']}</h1>
            <div class="description">{recipe_data['description']}</div>
            
            <div class="recipe-info">
                <span>Prep Time: {recipe_data['prepTime']}</span>
                <span>Cook Time: {recipe_data['cookTime']}</span>
                <span>Servings: {recipe_data['servings']}</span>
            </div>
            
            <div class="ingredients">
                <h2>Ingredients</h2>
                <ul>
                    {''.join(f'<li>{ingredient}</li>' for ingredient in recipe_data['ingredients'])}
                </ul>
            </div>
            
            <div class="instructions">
                <h2>Instructions</h2>
                <ol>
                    {''.join(f'<li>{instruction}</li>' for instruction in recipe_data['instructions'])}
                </ol>
            </div>
        </body>
        </html>
        """
        return html_template
    except Exception as e:
        logger.error(f"Error generating HTML: {str(e)}")
        raise

@app.route('/recipes/<filename>')
def custom_static(filename):
    return send_from_directory('recipes', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        logger.debug("Starting file upload processing")
        
        if 'photo' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['photo']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the image file
        image_data = file.read()
        
        # Get recipe from Gemini
        logger.debug("Generating recipe from image using Gemini")
        recipe_data = get_recipe_from_image(image_data)
        
        # Generate filename
        recipe_name = recipe_data["name"].replace(" ", "_").replace("/", "_").lower()
        filename = f"{recipe_name}.html"
        
        # Ensure recipes directory exists
        os.makedirs('recipes', exist_ok=True)
        filepath = os.path.join('recipes', filename)
        
        # Generate and save HTML
        logger.debug(f"Generating HTML for recipe: {recipe_name}")
        html_content = generate_html_from_recipe(recipe_data)
        
        # Add schema markup
        schema_json = json.dumps(recipe_data, indent=2)
        schema_html = f"<script type='application/ld+json'>{schema_json}</script>"
        full_html = f"{html_content}\n{schema_html}"
        
        # Save the file
        logger.debug(f"Saving recipe to {filepath}")
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(full_html)
        
        return jsonify({'message': 'Success', 'filepath': f'/recipes/{filename}'})
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)