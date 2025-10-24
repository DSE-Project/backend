# RecessionScope 📈

A comprehensive US Recession Forecasting System that provides 1, 3, and 6-month recession probability predictions using advanced machine learning models.

## 🌐 Live Demo

Visit the live application: **[https://recession-scope.vercel.app/](https://recession-scope.vercel.app/)**

## 📋 Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python 3.12+** (for backend)
- **Node.js 18+** and **npm** (for frontend)
- **Git** (for cloning repositories)
- **wkhtmltopdf** (for PDF generation)

## 🚀 Quick Setup Guide

### 1. Clone the Repositories

```bash
# Clone the backend repository
git clone https://github.com/DSE-Project/backend.git

# Clone the frontend repository  
git clone https://github.com/DSE-Project/frontend.git
```

### 2. Project Structure
After cloning, your directory structure should look like:
```
RecessionScope/
├── backend/
└── frontend/
```

---

## 🔧 Backend Setup

### Step 1: Navigate to Backend Directory
```bash
cd backend
```

### Step 2: Create Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```env
# Supabase Configuration (Contact developers for these values)
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# API Keys (Get your own)
GROQ_API_KEY=your_groq_api_key
FRED_API_KEY=your_fred_api_key

# PDF Generation
WKHTMLTOPDF_PATH=path_to_your_wkhtmltopdf_executable
```

#### Environment Variables Explained:

- **SUPABASE_URL & SUPABASE_ANON_KEY**: Database credentials for Supabase
  - 📧 **Contact us** to get these credentials
  
- **GROQ_API_KEY**: For AI/ML processing
  - Get your free API key from [Groq Console](https://console.groq.com/)
  
- **FRED_API_KEY**: For Federal Reserve Economic Data
  - Get your free API key: [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
  
- **WKHTMLTOPDF_PATH**: Path to wkhtmltopdf executable for PDF generation

### Step 3: Install wkhtmltopdf

Download and install wkhtmltopdf from: **[https://wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html)**

After installation, add the executable path to your `.env` file:

**Windows:**
```env
WKHTMLTOPDF_PATH="C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
```

**Linux:**
```env
WKHTMLTOPDF_PATH="/usr/local/bin/wkhtmltopdf"
```

**macOS:**
```env
WKHTMLTOPDF_PATH="/usr/local/bin/wkhtmltopdf"
```

### Step 4: Install Dependencies

#### Option A: Using UV Package Manager (Recommended)

UV is a fast Python package manager. Install it first:

**Windows:**
```powershell
# Using pip
pip install uv

# Or using winget
winget install astral-sh.uv
```

**macOS:**
```bash
# Using Homebrew
brew install uv

# Or using pip
pip install uv
```

**Linux:**
```bash
# Using pip
pip install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync dependencies:
```bash
uv sync
```

#### Option B: Using Traditional pip

```bash
# Create virtual environment
python -m venv venv

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Activate Virtual Environment

**Windows (Command Prompt/PowerShell):**
```powershell
# If using uv
.venv\Scripts\activate

# If using traditional venv
venv\Scripts\activate
```

**Unix-based (Linux/macOS):**
```bash
# If using uv
source .venv/bin/activate

# If using traditional venv
source venv/bin/activate
```

### Step 6: Run the Backend Server

```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or using the Python script
python main.py
```

**Backend will be available at:** `http://127.0.0.1:8000`

**API Documentation:** `http://127.0.0.1:8000/docs`

---

## 🎨 Frontend Setup

### Step 1: Navigate to Frontend Directory
```bash
cd frontend
```

### Step 2: Create Environment Configuration

Create a `.env` file in the frontend directory:

```env
# API Configuration
VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1

# Supabase Configuration (Same as backend - contact developers)
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Frontend URL
VITE_FRONTEND_URL=http://localhost:5173
```

#### Environment Variables Explained:

- **VITE_API_BASE_URL**: Backend API endpoint
  - Local development: `http://127.0.0.1:8000/api/v1`
  - Production backend: `https://recessionscope.duckdns.org/api/v1`
  
- **VITE_SUPABASE_URL & VITE_SUPABASE_ANON_KEY**: Same as backend credentials
  - 📧 **Contact us** to get these credentials
  
- **VITE_FRONTEND_URL**: Frontend application URL
  - Local development: `http://localhost:5173`

### Step 3: Install Dependencies

```bash
npm install
```

### Step 4: Run the Frontend Server

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

**Frontend will be available at:** `http://localhost:5173`

---

## 🔄 Full Development Workflow

1. **Start Backend Server:**
   ```bash
   cd backend
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start Frontend Server (in another terminal):**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access Application:**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://127.0.0.1:8000`
   - API Docs: `http://127.0.0.1:8000/docs`

---

## 📁 Project Architecture

### Backend Structure
```
backend/
├── api/v1/              # API endpoints
├── services/            # Business logic services
├── models/              # ML models (1m, 3m, 6m)
├── schemas/             # Data validation schemas
├── utils/               # Utility functions
├── config/              # Configuration files
└── middleware/          # Custom middleware
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/          # Page components
│   ├── api/            # API service functions
│   ├── contexts/       # React contexts
│   ├── hooks/          # Custom React hooks
│   └── utils/          # Utility functions
└── public/             # Static assets
```

---

## 🛠️ Troubleshooting

### Common Issues:

1. **Backend won't start:**
   - Check if Python 3.12+ is installed
   - Verify all environment variables are set
   - Ensure wkhtmltopdf is properly installed
   - Check if port 8000 is available

2. **Frontend won't start:**
   - Verify Node.js 18+ and npm are installed
   - Check if port 5173 is available
   - Ensure environment variables are correctly set

3. **API connection issues:**
   - Verify backend is running on `http://127.0.0.1:8000`
   - Check `VITE_API_BASE_URL` in frontend `.env`
   - Disable firewall/antivirus temporarily for testing

4. **PDF Generation not working:**
   - Verify wkhtmltopdf installation
   - Check `WKHTMLTOPDF_PATH` in backend `.env`
   - Ensure the path exists and is executable

### Getting Help:

- 📧 **Contact developers** for Supabase credentials
- 🐛 **Report issues** on the respective GitHub repositories
- 📖 **Check API documentation** at `http://127.0.0.1:8000/docs`

---

## 🔗 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [Supabase Documentation](https://supabase.com/docs)

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 👥 Contributors

Developed by the DSEP Group 18.

**Need help?** Contact us for Supabase credentials and additional support!
