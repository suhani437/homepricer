import { Home, Calculator, TrendingUp, BarChart3, Brain } from "lucide-react";
import { Link } from "wouter";
import PropertyForm from "@/components/property-form";
import PredictionResults from "@/components/prediction-results";
import FeatureImportance from "@/components/feature-importance";
import ModelMetrics from "@/components/model-metrics";
import MarketInsights from "@/components/market-insights";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Home className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                HomePricer
              </span>
            </div>
            <nav className="flex items-center space-x-6">
              <Link href="/" className="text-blue-600 font-medium">
                Home
              </Link>
              <Link href="/about" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">
                About
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent mb-4">
            Predict House Prices with ML!
          </h2>
          <p className="text-lg text-gray-700 max-w-2xl mx-auto">
            <span className="text-blue-600 font-semibold">Get accurate home price estimates</span> using our{" "}
            <span className="text-purple-600 font-semibold">linear regression model</span>. Presented by students of{" "}
            <span className="text-green-600 font-bold">BCA 'F1'</span>.
          </p>
        </div>



        {/* Main Prediction Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Form Section */}
          <div className="lg:col-span-2">
            <PropertyForm />
          </div>

          {/* Results Section */}
          <div className="lg:col-span-1">
            <PredictionResults />
          </div>
        </div>
      </main>

      {/* Credits Footer */}
      <footer className="bg-gradient-to-r from-blue-50 to-purple-50 border-t border-gray-200 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p className="text-gray-700 font-medium">
              © 2025 <span className="text-blue-600 font-bold">HomePricer</span>. Built by{" "}
              <span className="text-purple-600 font-semibold">Suhani Bora</span>,{" "}
              <span className="text-green-600 font-semibold">Shivanshi Rawat</span>,{" "}
              <span className="text-orange-600 font-semibold">Mahak Santokhi</span>, and{" "}
              <span className="text-pink-600 font-semibold">Saniya Bisht</span>.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
