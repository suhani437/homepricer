# HomePricer Deployment Guide

This guide helps you deploy HomePricer to various platforms for your academic project presentation.

## 📋 Pre-Deployment Checklist

- [ ] All code committed to GitHub
- [ ] Dependencies listed in package.json
- [ ] Environment variables documented
- [ ] Build process tested locally
- [ ] Python dependencies available

## 🚀 Deployment Options

### Option 1: Replit (Recommended for Students)

**Pros:** Free, easy setup, automatic Python environment
**Best for:** Academic presentations, quick demos

1. **Import to Replit:**
   - Go to [replit.com](https://replit.com)
   - Click "Create Repl" → "Import from GitHub"
   - Paste your repository URL
   - Select Node.js template

2. **Configure Replit:**
   - Replit auto-detects the configuration
   - Click "Run" button
   - Share the live URL with professors/classmates

3. **Access your app:**
   ```
   https://YOUR-REPL-NAME.YOUR-USERNAME.repl.co
   ```

### Option 2: Vercel (Professional Deployment)

**Pros:** Fast, free tier, automatic deployments
**Best for:** Portfolio projects, professional presentation

1. **Connect GitHub:**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub account
   - Click "New Project" → Select your repository

2. **Configure Build:**
   ```
   Framework Preset: Vite
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

3. **Deploy:**
   - Click "Deploy"
   - Get your live URL: `https://homepricer.vercel.app`

### Option 3: Netlify

**Pros:** Free hosting, easy drag-and-drop
**Best for:** Static frontend deployment

1. **Build locally:**
   ```bash
   npm run build
   ```

2. **Deploy to Netlify:**
   - Go to [netlify.com](https://netlify.com)
   - Drag `dist/` folder to deploy area
   - Get instant live URL

3. **For full-stack:**
   - Use Netlify Functions for backend
   - Configure serverless Python functions

### Option 4: Railway

**Pros:** Full-stack support, free tier
**Best for:** Complete application with backend

1. **Connect GitHub:**
   - Go to [railway.app](https://railway.app)
   - Connect GitHub repository
   - Select "Deploy from GitHub repo"

2. **Auto-deployment:**
   - Railway detects Node.js automatically
   - Installs Python dependencies
   - Provides public URL

## 🔧 Local Development Setup

For professors or classmates to run locally:

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/homepricer.git
cd homepricer

# Install dependencies
npm install

# Start development server
npm run dev

# Open browser
# http://localhost:5000
```

## 📱 Mobile Access

All deployment options provide mobile-responsive URLs that work on:
- Smartphones
- Tablets  
- Desktop browsers
- Classroom projectors

## 🎯 For Academic Presentation

### Quick Demo Setup (2 minutes):
1. Open deployed URL
2. Show Home page with prediction form
3. Enter sample property details
4. Demonstrate live price prediction
5. Navigate to About page for project details

### Sample Demo Data:
```
Square Footage: 1800
Year Built: 2010
Bedrooms: 3
Bathrooms: 2
Garage: 2
Property Type: Single Family
```
Expected output: ~₹65-75 Lakhs

## 🔍 Troubleshooting

### Common Issues:

**Build Fails:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Python Model Not Working:**
- Ensure Python 3.11+ available
- Install scikit-learn, numpy, pandas
- Check file paths in deployment environment

**API Errors:**
- Verify backend routes are accessible
- Check CORS configuration
- Ensure Python subprocess works in deployment environment

### Environment Variables:
```bash
# If needed for production
NODE_ENV=production
PYTHON_PATH=python3
PORT=5000
```

## 📊 Performance Optimization

For faster loading:
- Images optimized as SVG
- Minimal dependencies
- Code splitting enabled
- Static file caching

## 🎓 Academic Submission

Include these URLs in your project report:
- **Live Demo:** [Your deployment URL]
- **Source Code:** https://github.com/YOUR-USERNAME/homepricer
- **Documentation:** README.md and this deployment guide

## 📞 Support

For deployment issues:
- Check platform documentation
- Review build logs
- Test locally first
- Verify all dependencies installed

---

**Ready to deploy? Choose Option 1 (Replit) for the easiest academic presentation setup.**