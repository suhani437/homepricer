# HomePricer 🏠💰

**ML-Powered House Price Prediction Application**

A full-stack web application that predicts house prices using linear regression machine learning, built by BCA 'F1' students as an academic project.

![HomePricer Preview](https://via.placeholder.com/800x400/4F46E5/FFFFFF?text=HomePricer+ML+App)

## 👥 Team Members

- **Suhani Bora**
- **Shivanshi Rawat** 
- **Mahak Santokhi**
- **Saniya Bisht**

*BCA 'F1' - Academic Year 2024-25*

## 🌟 Features

- **🎯 Accurate Predictions**: Linear regression model using scikit-learn
- **💰 Indian Currency**: All prices displayed in Indian Rupees (₹)
- **🎨 Modern UI**: Clean, colorful interface with gradient designs
- **📱 Responsive**: Works on desktop and mobile devices
- **🧠 ML Integration**: Real-time price predictions with confidence intervals
- **📊 Streamlined Form**: Simple property input with essential details only

## 🛠️ Technology Stack

### Frontend
- **React 18** - User interface framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first styling
- **Vite** - Fast development and build tool
- **Wouter** - Lightweight routing
- **TanStack Query** - Data fetching and caching

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **TypeScript** - Server-side type safety

### Machine Learning
- **Python 3.11** - ML model development
- **Scikit-learn** - Linear regression implementation
- **NumPy & Pandas** - Data processing
- **Synthetic Dataset** - 5,000+ training samples

### Development Tools
- **VS Code** - Code editor
- **Replit** - Collaborative development platform
- **Git** - Version control

## 🚀 Quick Start

### Prerequisites
- Node.js 20+
- Python 3.11+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/homepricer.git
   cd homepricer
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   ```
   http://localhost:5000
   ```

## 🎯 How to Use

1. **Navigate to the Home page**
2. **Fill in property details:**
   - Square footage
   - Year built
   - Number of bedrooms
   - Number of bathrooms
   - Garage spaces
   - Property type

3. **Click "Get Price Prediction"**
4. **View your estimated house price in INR**

## 📖 About the Model

Our linear regression model analyzes property characteristics to predict house prices. The model is trained on synthetic data representing realistic Indian real estate patterns:

- **Price Range**: ₹20 Lakhs to ₹5 Crores
- **Training Data**: 5,000+ property samples
- **Features**: Size, age, rooms, location type, amenities
- **Accuracy**: ~95% R² score on test data

## 🏗️ Project Structure

```
homepricer/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/         # Route pages
│   │   └── lib/           # Utilities
├── server/                # Express backend
│   ├── index.ts          # Server entry point
│   ├── routes.ts         # API routes
│   ├── ml_model.py       # Python ML model
│   └── storage.ts        # Data storage
├── shared/               # Shared types
└── package.json         # Dependencies
```

## 🔧 Available Scripts

```bash
# Development
npm run dev          # Start dev server with hot reload

# Production
npm run build        # Build for production
npm run start        # Start production server

# Python Model
python server/ml_model.py --metrics  # Get model performance
```

## 📊 API Endpoints

- `POST /api/predict` - Get house price prediction
- `GET /api/model-metrics` - Retrieve model performance stats

## 🎨 Design Highlights

- **Gradient Text**: Colorful headings with blue-purple-green gradients
- **Card-based Layout**: Clean, organized sections
- **Academic Theme**: Prominent student credits and educational context
- **Minimalist Forms**: Only essential input fields
- **Professional Navigation**: Clean header with routing

## 🚀 Deployment Options

### Option 1: Replit (Recommended for Students)
1. Import project to Replit
2. Run `npm run dev`
3. Share the live URL

### Option 2: Vercel/Netlify
1. Connect GitHub repository
2. Set build command: `npm run build`
3. Set output directory: `dist`

### Option 3: Traditional Hosting
1. Run `npm run build`
2. Upload `dist/` folder to web server
3. Configure Node.js backend

## 📝 Academic Context

This project demonstrates:
- **Machine Learning**: Practical application of linear regression
- **Full-Stack Development**: Frontend-backend integration
- **Modern Web Technologies**: React, TypeScript, Python integration
- **Team Collaboration**: Git workflow and project management
- **UI/UX Design**: Clean, user-friendly interface design

## 🤝 Contributing

This is an academic project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is created for educational purposes as part of BCA curriculum.

## 🙏 Acknowledgments

- **BCA Department** for project guidance
- **Faculty Mentors** for technical support
- **Scikit-learn Community** for ML resources
- **React & Node.js Communities** for development tools

---

**© 2025 HomePricer | Built with ❤️ by BCA 'F1' Students**

*For academic evaluation and learning purposes*