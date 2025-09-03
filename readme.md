# RecessionScope API

A FastAPI-based REST API for predicting US recession probabilities using machine learning models. The system provides 1-month, 3-month, and 6-month recession probability forecasts.

## Features

- **Multiple Time Horizons**: Get recession probability predictions for 1, 3, and 6 months
- **Machine Learning Models**: Powered by LSTM neural networks trained on economic indicators
- **RESTful API**: Clean and well-documented endpoints
- **Interactive Documentation**: Built-in Swagger UI and ReDoc documentation
- **CORS Support**: Ready for frontend integration

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

### Development Mode

To run the server in development mode with auto-reload:

```bash
fastapi dev main.py
```

The server will start on `http://localhost:8000`

### Production Mode

To run the server in production mode:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, you can access:

- **Interactive API Documentation (Swagger UI)**: http://localhost:8000/docs
- **Alternative Documentation (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### Main Endpoints

- `POST /api/v1/forecast/predict/all` - Get predictions from all models (1m, 3m, 6m)
- `POST /api/v1/forecast/predict/1m` - Get 1-month recession probability
- `POST /api/v1/forecast/predict/3m` - Get 3-month recession probability  
- `POST /api/v1/forecast/predict/6m` - Get 6-month recession probability

### Utility Endpoints

- `GET /` - Welcome message and API information
- `GET /health` - Health check endpoint

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── api/
│   └── v1/
│       └── forecast.py     # API route definitions
├── services/               # Business logic and model services
├── schemas/                # Pydantic data models
├── ml_models/              # Trained machine learning models
└── notebooks/              # Jupyter notebooks for data analysis
```

## Development

### Adding New Features

1. **Models**: Add new model files to `ml_models/`
2. **Schemas**: Define input/output data structures in `schemas/`
3. **Services**: Implement business logic in `services/`
4. **Routes**: Add API endpoints in `api/v1/`

### Testing

Run the development server and test endpoints using the interactive documentation at `/docs`.

## Environment Variables

You can configure the following environment variables:

- `DEBUG`: Set to `false` for production mode (default: `true`)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins (default: `*`)

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Make sure you're running commands from the `backend` directory
2. **Missing Dependencies**: Run `pip install -r requirements.txt` again
3. **Port Already in Use**: Change the port in `main.py` or kill the process using port 8000

### Model Loading Issues

If you encounter model loading errors:
- Ensure model files exist in the `ml_models/` directory
- Check that model files are in the correct format (.keras)
- Verify TensorFlow compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.