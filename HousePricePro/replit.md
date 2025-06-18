# HomePricer - ML-Powered House Price Prediction

## Overview

HomePricer is a full-stack web application that predicts house prices using machine learning, developed by BCA 'F1' students: Suhani Bora, Shivanshi Rawat, Mahak Santokhi, and Saniya Bisht. The system combines a React frontend with an Express.js backend and a Python linear regression model to provide accurate house price estimates in Indian Rupees (INR).

## System Architecture

The application follows a modern full-stack architecture with clear separation of concerns:

- **Frontend**: React with TypeScript, using Vite for development and building
- **Backend**: Express.js server with TypeScript
- **Database**: PostgreSQL with Drizzle ORM for data persistence
- **ML Model**: Python-based linear regression model using scikit-learn
- **UI Framework**: Tailwind CSS with shadcn/ui components
- **State Management**: TanStack Query for server state management

## Key Components

### Frontend Architecture
- Built with React 18 and TypeScript
- Uses Vite for fast development and optimized builds
- shadcn/ui component library for consistent UI design
- TanStack Query for efficient data fetching and caching
- Wouter for lightweight client-side routing
- React Hook Form with Zod validation for form handling

### Backend Architecture
- Express.js server with TypeScript
- RESTful API design with clear endpoint structure
- Middleware for request logging and error handling
- Python subprocess integration for ML model execution
- In-memory storage implementation (can be extended to use PostgreSQL)

### Data Storage
- PostgreSQL database configured with Drizzle ORM
- Two main tables: `properties` and `predictions`
- Type-safe database operations with generated TypeScript types
- Migration support through Drizzle Kit

### Machine Learning Model
- Python-based linear regression model using scikit-learn
- Trained on synthetic real estate data (5000+ samples)
- Features include: square footage, year built, bedrooms, bathrooms, garage, property type, neighborhood, and boolean features
- Provides prediction confidence intervals and feature importance analysis

## Data Flow

1. **User Input**: User fills out property form in React frontend
2. **Form Validation**: Client-side validation using Zod schema
3. **API Request**: Form data sent to Express backend via REST API
4. **ML Prediction**: Backend spawns Python process to generate price prediction
5. **Data Storage**: Property data and prediction results stored in database
6. **Response**: Prediction results returned to frontend and displayed to user
7. **State Management**: TanStack Query manages caching and updates

## External Dependencies

### Frontend Dependencies
- React ecosystem (React, React DOM, React Hook Form)
- UI components (@radix-ui/* packages for headless components)
- Styling (Tailwind CSS, class-variance-authority, clsx)
- Data fetching (@tanstack/react-query)
- Date handling (date-fns)
- Form validation (zod, @hookform/resolvers)

### Backend Dependencies
- Express.js with TypeScript support
- Database ORM (drizzle-orm, @neondatabase/serverless)
- Development tools (tsx for TypeScript execution)
- Build tools (esbuild for production builds)

### Python Dependencies
- Machine learning (scikit-learn, numpy, pandas)
- Standard Python libraries for data processing

## Deployment Strategy

The application is configured for deployment on Replit with the following setup:

- **Development**: `npm run dev` starts both Vite dev server and Express backend
- **Production Build**: `npm run build` creates optimized frontend bundle and compiles backend
- **Production Start**: `npm run start` runs the compiled backend serving static frontend files
- **Port Configuration**: Backend runs on port 5000, exposed as port 80 externally
- **Database**: PostgreSQL 16 configured and ready for connection
- **Environment**: Node.js 20 with Python 3.11 support

The build process:
1. Frontend built with Vite to `dist/public`
2. Backend compiled with esbuild to `dist/index.js`
3. Static file serving configured for production deployment

## Key Features

- **Clean Interface**: Streamlined design with colorful gradient headings and minimal form fields
- **Indian Currency**: All price predictions shown in Indian Rupees (₹)
- **Navigation**: Professional header with Home and About pages
- **Student Project**: Prominently features BCA 'F1' team credits and academic context
- **Simplified Form**: Only essential property details (size, year, rooms, type)
- **Linear Regression**: Uses scikit-learn for house price prediction
- **Responsive Design**: Works across desktop and mobile devices

## Final Implementation Details

### Form Fields
- Square Footage
- Year Built  
- Bedrooms (1-5+)
- Bathrooms (1-4+)
- Garage Spaces (0-3+)
- Property Type (Single Family, Condo, Townhouse, Multi-Family)

### Removed Features (per user requests)
- Additional features section (pool, fireplace, etc.)
- Neighborhood selection
- Model performance metrics display
- Feature importance visualization
- Market insights section
- All decorative images and logos

### Pages
- **Home**: Main prediction interface with property form and results
- **About**: Comprehensive project information, technology stack, and team credits

## Changelog

- June 16, 2025: Initial setup and ML model implementation
- June 16, 2025: Updated to HomePricer with Indian pricing, simplified interface, and navigation
- June 16, 2025: Final version with student credits and About page

## User Preferences

Preferred communication style: Simple, everyday language.
Project focus: Student academic project for BCA 'F1' with colorful, clean design.