from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import base64
import os
import io
import json
import logging
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_class_labels():
    try:
        train_path = r"C:/Users/ashok/Downloads/Indian_Food_Images/Indian_Food_Images"
        class_labels = sorted(os.listdir(train_path))  
        logger.info(f"Loaded {len(class_labels)} class labels")
        return class_labels
    except Exception as e:
        logger.error(f"Error loading class labels: {str(e)}")
        return []

class CustomEfficientNet(torch.nn.Module):
    def __init__(self, num_classes=206):  
        super(CustomEfficientNet, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.classifier = torch.nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model():
    try:
        model = CustomEfficientNet().to(device)
        model.load_state_dict(torch.load('model_weights.pth', map_location=device))
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = load_model()
class_labels = load_class_labels()

def predict_with_model(image):
    try:
        if not class_labels:
            logger.error("No class labels available")
            return None, 0.0
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            
            confidence = max_prob.item()
            predicted_class = class_labels[predicted.item()]
            
            logger.info(f"Model predicted {predicted_class} with confidence {confidence}")
            return predicted_class, confidence
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return None, 0.0

def get_gemini_endpoint():
    return f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def encode_image_to_base64(image_data):
    try:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise

def get_recipe_from_image(image_data, predicted_class=None):
    try:
        base64_image = encode_image_to_base64(image_data)
        
        prompt = """
        Look at this food image and:
        """
        if predicted_class:
            prompt += f"""
            This image has been identified as {predicted_class}, which is an Indian dish.
            Generate a detailed recipe for {predicted_class} including:
            """
        else:
            prompt += """
            1. Identify the Indian dish.
            2. Generate a detailed recipe for it including:
            """
        
        prompt += """
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

        headers = {'Content-Type': 'application/json'}
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

        response = requests.post(get_gemini_endpoint(), headers=headers, json=data)
        response.raise_for_status()
        
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
        model_info = ""
        if "model_prediction" in recipe_data:
            pred = recipe_data["model_prediction"]
            model_info = f"""
            <div class="model-info">
                <h3>Model Prediction</h3>
                <p>Predicted Class: {pred['predicted_class']}</p>
                <p>Confidence: {pred['confidence']}%</p>
            </div>
            """

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
                .model-info {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
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
            
            {model_info}
            
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
        image = Image.open(io.BytesIO(image_data))
        
        # Try model prediction first
        predicted_class = None
        confidence = 0.0
        
        if model is not None:
            predicted_class, confidence = predict_with_model(image)
            logger.debug(f"Model prediction: {predicted_class} with confidence: {confidence}")
        
        # Use model prediction if confidence is high enough, otherwise let Gemini identify
        use_prediction = predicted_class is not None and confidence > 0.7
        recipe_data = get_recipe_from_image(image_data, predicted_class if use_prediction else None)
        
        # Generate filename and save HTML
        recipe_name = recipe_data["name"].replace(" ", "_").replace("/", "_").lower()
        filename = f"{recipe_name}.html"
        
        os.makedirs('recipes', exist_ok=True)
        filepath = os.path.join('recipes', filename)
        
        # Add model prediction info to recipe data if available
        if predicted_class and confidence > 0:
            recipe_data["model_prediction"] = {
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2)
            }
        
        # Generate and save HTML
        html_content = generate_html_from_recipe(recipe_data)
        schema_json = json.dumps(recipe_data, indent=2)
        schema_html = f"<script type='application/ld+json'>{schema_json}</script>"
        full_html = f"{html_content}\n{schema_html}"
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(full_html)
        
        return jsonify({
            'message': 'Success',
            'filepath': f'/recipes/{filename}',
            'model_prediction': {
                'class': predicted_class,
                'confidence': round(confidence * 100, 2)
            } if predicted_class else None
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recipes/<filename>')
def custom_static(filename):
    return send_from_directory('recipes', filename)

if __name__ == '__main__':
    app.run(debug=True)