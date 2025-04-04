from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import logging
import json
from functools import wraps
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize cache
roadmap_cache = {}

DJANGO_API_URL = "http://127.0.0.1:8000/projects/save_project/"

def validate_request(f):
    """Decorator to validate request data"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' field in request"}), 400
            
        if not isinstance(data['prompt'], str) or len(data['prompt'].strip()) == 0:
            return jsonify({"error": "Prompt must be a non-empty string"}), 400
            
        return f(*args, **kwargs)
    return decorated_function

def validate_auth(f):
    """Decorator to validate authentication token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Token '):
            return jsonify({"error": "Missing or invalid authentication token"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def get_prompt_template(prompt):
    """Get the prompt template for generating roadmap"""
    return f"""Create a detailed project roadmap for: {prompt}

    Return the response in this exact JSON format without any additional text or markdown:
    {{
        "project_overview": {{
            "name": "Project Name",
            "description": "Comprehensive project description",
            "estimated_duration": "Total estimated timeline",
            "team_size": "Recommended team size",
            "objectives": [
                "Clear objective 1 with measurable outcomes",
                "Clear objective 2 with specific goals",
                "Clear objective 3 with defined targets"
            ],
            "success_criteria": [
                "Specific success metric 1",
                "Specific success metric 2",
                "Specific success metric 3"
            ]
        }},
        "phases": {{
            "phase1": {{
                "name": "Phase Name",
                "overview": "Detailed phase description",
                "duration": "Phase duration (e.g., 2-3 weeks)",
                "objectives": [
                    "Phase-specific objective 1",
                    "Phase-specific objective 2"
                ],
                "deliverables": [
                    "Concrete deliverable 1",
                    "Concrete deliverable 2"
                ],
                "sub_phases": {{
                    "sub1": {{
                        "name": "Sub-phase Name",
                        "description": "Detailed sub-phase description",
                        "tasks": [
                            "Specific task 1 with clear definition",
                            "Specific task 2 with acceptance criteria",
                            "Specific task 3 with dependencies"
                        ],
                        "deliverables": [
                            "Sub-phase deliverable 1",
                            "Sub-phase deliverable 2"
                        ],
                        "estimated_time": "Sub-phase duration"
                    }},
                    "sub2": {{
                        "name": "Another Sub-phase",
                        "description": "Another detailed description",
                        "tasks": [
                            "More specific tasks with clear outcomes",
                            "Technical requirements and specifications",
                            "Integration points and dependencies"
                        ],
                        "deliverables": [
                            "More concrete deliverables",
                            "Specific outputs and artifacts"
                        ],
                        "estimated_time": "Sub-phase duration"
                    }}
                }},
                "dependencies": [
                    "Dependency 1",
                    "Dependency 2"
                ],
                "risks": [
                    "Potential risk 1 with mitigation strategy",
                    "Potential risk 2 with contingency plan"
                ],
                "milestones": [
                    "Key milestone 1 with completion criteria",
                    "Key milestone 2 with verification method"
                ]
            }}
        }}
    }}

    Important instructions:
    1. Create at least 6 main phases
    2. Each main phase should have 2-4 sub-phases
    3. Provide detailed, specific tasks for each sub-phase
    4. Include realistic time estimates
    5. Ensure dependencies between phases are logical
    6. Add meaningful milestones for each phase
    7. Include comprehensive risk assessment
    """

def generate_roadmap(prompt):
    """Generate a structured project roadmap using Gemini"""
    try:
        # Check cache first
        cache_key = prompt.lower().strip()
        if cache_key in roadmap_cache:
            logger.info(f"Returning cached roadmap for prompt: {prompt}")
            return roadmap_cache[cache_key]

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        logger.info(f"Initialized model: gemini-1.5-pro")

        # Get response from model with settings for detailed output
        response = model.generate_content(
            get_prompt_template(prompt),
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=1,
                top_k=32,
                max_output_tokens=8192
            )
        )
        
        if not response or not response.text:
            logger.error("No response generated from model")
            return create_default_roadmap(prompt)

        try:
            # Log the raw response for debugging
            logger.info(f"Raw response from Gemini: {response.text}")
            
            # Clean and parse the response
            response_text = response.text.strip()
            
            # Try to find JSON structure
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                
                # Add detailed logging of JSON string
                logger.info(f"Attempting to parse JSON string: {json_str[:200]}...")
                
                try:
                    # First try direct JSON parsing
                    roadmap = json.loads(json_str)
                    
                    # Validate required fields
                    if not isinstance(roadmap, dict):
                        logger.error("Response is not a dictionary")
                        return create_default_roadmap(prompt)
                        
                    if 'project_overview' not in roadmap or 'phases' not in roadmap:
                        logger.error("Missing required top-level fields")
                        return create_default_roadmap(prompt)
                        
                    # Validate project overview
                    overview = roadmap['project_overview']
                    required_overview_fields = ['name', 'description', 'estimated_duration', 'team_size']
                    if not all(field in overview for field in required_overview_fields):
                        logger.error("Missing required project overview fields")
                        return create_default_roadmap(prompt)
                        
                    # Validate phases
                    phases = roadmap['phases']
                    if not isinstance(phases, dict) or len(phases) < 1:
                        logger.error("Invalid phases structure")
                        return create_default_roadmap(prompt)
                        
                    # Add metadata
                    roadmap['metadata'] = {
                        'generated_at': datetime.now().isoformat(),
                        'prompt': prompt,
                        'version': '2.0'
                    }
                    
                    # Cache the result
                    roadmap_cache[cache_key] = roadmap
                    logger.info(f"Successfully generated and validated roadmap for prompt: {prompt}")
                    return roadmap
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {str(e)}")
                    logger.error(f"Error location: Line {e.lineno}, Column {e.colno}")
                    
                    # Try cleaning the JSON string
                    try:
                        # Remove any markdown backticks or extra whitespace
                        clean_json = json_str.replace('```json', '').replace('```', '').strip()
                        
                        # Add detailed logging of cleaned JSON
                        logger.info(f"Attempting to parse cleaned JSON: {clean_json[:200]}...")
                        
                        roadmap = json.loads(clean_json)
                        
                        # Add metadata
                        roadmap['metadata'] = {
                            'generated_at': datetime.now().isoformat(),
                            'prompt': prompt,
                            'version': '2.0'
                        }
                        
                        # Cache the result
                        roadmap_cache[cache_key] = roadmap
                        logger.info(f"Successfully parsed cleaned JSON for prompt: {prompt}")
                        return roadmap
                    except Exception as clean_error:
                        logger.error(f"Failed to parse cleaned JSON: {str(clean_error)}")
                        if hasattr(clean_error, 'lineno') and hasattr(clean_error, 'colno'):
                            logger.error(f"Error location: Line {clean_error.lineno}, Column {clean_error.colno}")
                        return create_default_roadmap(prompt)
            else:
                logger.error("No JSON structure found in response")
                return create_default_roadmap(prompt)
                
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return create_default_roadmap(prompt)
            
    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        return create_default_roadmap(prompt)

def create_default_roadmap(prompt):
    """Create a default roadmap structure when the API response fails"""
    return {
        "project_overview": {
            "name": f"{prompt.title()} Project",
            "description": f"A comprehensive project plan for {prompt}, focusing on delivering a high-quality solution that meets all requirements while maintaining best practices in development, testing, and deployment.",
            "estimated_duration": "12 weeks",
            "team_size": "4-6 people",
            "objectives": [
                "Deliver a robust and scalable solution that meets all functional requirements",
                "Ensure high code quality and comprehensive test coverage",
                "Implement modern best practices and design patterns",
                "Create detailed documentation for maintenance and future development",
                "Establish efficient CI/CD pipeline for seamless deployment"
            ],
            "success_criteria": [
                "Successful deployment to production",
                "High customer satisfaction",
                "Low defect rate",
                "Improved team velocity"
            ]
        },
        "phases": {
            "phase1": {
                "name": "Project Initiation & Planning",
                "overview": "Comprehensive project setup and detailed planning phase to establish strong foundations",
                "duration": "2-3 weeks",
                "objectives": [
                    "Define project scope and objectives",
                    "Develop detailed project schedule and timeline"
                ],
                "deliverables": [
                    "Project plan document",
                    "Project schedule and timeline"
                ],
                "sub_phases": {
                    "sub1": {
                        "name": "Requirements Analysis",
                        "description": "Deep dive into project requirements and technical specifications",
                        "tasks": [
                            "Conduct stakeholder interviews for requirement gathering",
                            "Document functional and non-functional requirements",
                            "Create detailed technical specifications",
                            "Define project success criteria and metrics",
                            "Identify potential risks and mitigation strategies"
                        ],
                        "deliverables": [
                            "Detailed project plan and timeline",
                            "Technical specification document",
                            "Development environment setup guide",
                            "Project coding standards document"
                        ],
                        "estimated_time": "1-2 weeks"
                    },
                    "sub2": {
                        "name": "Project Setup",
                        "description": "Setting up development infrastructure and tooling",
                        "tasks": [
                            "Configure development environment with necessary tools",
                            "Set up version control system and branching strategy",
                            "Initialize project structure with best practices",
                            "Configure linting and code formatting tools",
                            "Set up project management and tracking tools"
                        ],
                        "deliverables": [
                            "Development environment setup guide",
                            "Version control system setup guide",
                            "Project structure and organization guide"
                        ],
                        "estimated_time": "1 week"
                    }
                },
                "dependencies": [],
                "risks": ["risk1", "risk2"],
                "milestones": ["milestone1", "milestone2"]
            }
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/roadmap', methods=['POST'])
@validate_request
@validate_auth
def get_roadmap():
    """Generate a roadmap based on the prompt and save it to Django backend"""
    try:
        data = request.get_json()
        prompt = data['prompt']
        
        # Get the roadmap
        roadmap = generate_roadmap(prompt)
        if not roadmap:
            return jsonify({"error": "Failed to generate roadmap"}), 500

        # Extract project name from the roadmap
        project_name = roadmap["project_overview"]["name"]

        # Get the auth token from request header
        auth_token = request.headers.get('Authorization').split(' ')[1]

        # Save to Django backend with user's token
        django_headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
        
        django_data = {
            'name': project_name,
            'roadmap': roadmap
        }
        
        response = requests.post(
            DJANGO_API_URL,
            headers=django_headers,
            json=django_data
        )
        
        if response.status_code != 201:
            logger.error(f"Failed to save project to Django: {response.text}")
            return jsonify({"error": "Failed to save project"}), response.status_code

        return jsonify({
            "message": "Roadmap generated and saved successfully",
            "roadmap": roadmap
        })

    except Exception as e:
        logger.error(f"Error in get_roadmap: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the roadmap cache"""
    try:
        roadmap_cache.clear()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting the Flask application...")
    app.run(debug=True)