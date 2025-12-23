# BA CreateX API

A comprehensive FastAPI-based web service for image processing and personalized story generation with character integration.

## Features

### Text with Image Service
- **Personalized Story Generation**: Create custom stories with user-uploaded character images
- **Multi-language Support**: Generate stories in English, Spanish, French, Italian, and Arabic
- **Flexible Story Lengths**: Choose from single page to 10-chapter stories
- **Artistic Styles**: Multiple visual styles including Cartoon, Storybook, Illustration, Colorful, and Simple
- **Character Integration**: Upload character images that appear throughout the generated story
- **Age-appropriate Content**: Customize stories based on character age and gender

#### Story Generation Features:
- **Character Customization**: Set gender (Male/Female), name, and age (1-100)
- **Style Selection**: Choose from 5 different artistic styles for story illustrations
- **Language Options**: Generate stories in 5 different languages with proper cultural adaptation
- **Story Ideas**: Input custom story concepts (10-1000 characters)
- **Chapter Control**: Select story length from single page to 10 chapters
- **Image Processing**: Upload character images that become the story's main character

### Image Converter Service
- Convert between multiple image formats (JPEG, PNG, WEBP, BMP, TIFF, GIF)
- Image resizing with aspect ratio preservation
- Quality control for lossy formats
- Image information extraction

## Project Structure

```
├── app/
│   └── services/
│       ├── Text_with_image/
│       │   ├── Text_with_image.py          # Service logic
│       │   ├── Text_with_image_Route.py    # API routes
│       │   └── Text_with_image_Schema.py   # Pydantic schemas
│       └── Image_converter/
│           ├── Image_converter.py          # Service logic
│           ├── Image_converter_Route.py    # API routes
│           └── Image_converter_Schema.py   # Pydantic schemas
├── main.py                                 # FastAPI application
├── requirements.txt                        # Python dependencies
├── Dockerfile                             # Docker configuration
├── docker-compose.yml                     # Docker Compose setup
└── README.md                              # This file
```

## Installation

### Local Development

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ba_createx
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python main.py
```

The API will be available at `http://localhost:8065`

### Docker Deployment

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

2. **Run only the API service:**
```bash
docker-compose up image-processing-api
```

3. **Run with nginx proxy:**
```bash
docker-compose --profile with-proxy up
```

## API Documentation

Once the application is running, you can access:
- **Swagger UI:** `http://localhost:8065/docs`
- **ReDoc:** `http://localhost:8065/redoc`

## API Endpoints

### Text with Image Service

#### Add Text to Image
- **POST** `/api/v1/text-with-image/add-text`
- Add text overlay to an image

**Request Body:**
```json
{
    "image_base64": "base64-encoded-image-data",
    "text": "Your text here",
    "position_x": 10,
    "position_y": 10,
    "font_size": 20,
    "font_color_r": 255,
    "font_color_g": 255,
    "font_color_b": 255,
    "font_path": null
}
```

#### Health Check
- **GET** `/api/v1/text-with-image/health`

### Image Converter Service

#### Convert Image Format
- **POST** `/api/v1/image-converter/convert`
- Convert image to different format

**Request Body:**
```json
{
    "image_base64": "base64-encoded-image-data",
    "target_format": "JPEG",
    "quality": 85
}
```

#### Resize Image
- **POST** `/api/v1/image-converter/resize`
- Resize an image

**Request Body:**
```json
{
    "image_base64": "base64-encoded-image-data",
    "width": 800,
    "height": 600,
    "maintain_aspect_ratio": true
}
```

#### Get Image Information
- **POST** `/api/v1/image-converter/info`
- Get detailed information about an image

#### Get Supported Formats
- **GET** `/api/v1/image-converter/supported-formats`

#### Health Check
- **GET** `/api/v1/image-converter/health`

### General Endpoints

#### Root
- **GET** `/`
- API information and available endpoints

#### Health Check
- **GET** `/health`
- Overall API health status

## Usage Examples

### Adding Text to Image (Python)

```python
import requests
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# API request
response = requests.post(
    "http://localhost:8065/api/v1/text-with-image/add-text",
    json={
        "image_base64": image_data,
        "text": "Hello World!",
        "position_x": 50,
        "position_y": 50,
        "font_size": 30,
        "font_color_r": 255,
        "font_color_g": 0,
        "font_color_b": 0
    }
)

# Save result
result = response.json()
if result["success"]:
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(result["image_base64"]))
```

### Converting Image Format (Python)

```python
import requests
import base64

# Read and encode image
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Convert to JPEG
response = requests.post(
    "http://localhost:8065/api/v1/image-converter/convert",
    json={
        "image_base64": image_data,
        "target_format": "JPEG",
        "quality": 90
    }
)

# Save result
result = response.json()
if result["success"]:
    with open("converted.jpg", "wb") as f:
        f.write(base64.b64decode(result["image_base64"]))
```

## Development

### Adding New Features

1. Create service logic in the appropriate service directory
2. Define Pydantic schemas for request/response models
3. Implement API routes using FastAPI
4. Add the router to `main.py`
5. Update documentation

### Testing

Run tests with pytest:
```bash
pytest
```

## Configuration

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)

### Docker Configuration

The application includes health checks and runs as a non-root user for security.

## Supported Image Formats

- **Input:** JPEG, PNG, WEBP, BMP, TIFF, GIF
- **Output:** JPEG, PNG, WEBP, BMP, TIFF, GIF

## Error Handling

The API includes comprehensive error handling with appropriate HTTP status codes:
- 400: Bad Request (invalid input)
- 404: Not Found
- 500: Internal Server Error

## Security Considerations

- CORS is configured (update for production)
- Runs as non-root user in Docker
- Input validation using Pydantic
- No file system access (uses memory-based operations)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]