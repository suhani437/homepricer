import { Home, Code, Database, Brain, Globe } from "lucide-react";
import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AboutPage() {
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
              <Link href="/" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">
                Home
              </Link>
              <Link href="/about" className="text-blue-600 font-medium">
                About
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent mb-4">
            About HomePricer
          </h1>
          <p className="text-lg text-gray-700 max-w-3xl mx-auto leading-relaxed">
            House Pricer is a mini project developed by a group of four BCA students from{" "}
            <span className="text-green-600 font-bold">'F1'</span>—{" "}
            <span className="text-purple-600 font-semibold">Suhani Bora</span>,{" "}
            <span className="text-blue-600 font-semibold">Shivanshi Rawat</span>,{" "}
            <span className="text-orange-600 font-semibold">Mahak Santokhi</span>, and{" "}
            <span className="text-pink-600 font-semibold">Saniya Bisht</span>. The aim of this project is to predict house prices using{" "}
            <span className="text-purple-600 font-semibold">linear regression</span>, a fundamental machine learning technique.
          </p>
        </div>

        {/* Technology Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center space-x-3">
              <Code className="h-6 w-6 text-blue-600" />
              <span className="text-blue-600">Our Technology</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-600 rounded-full mt-2"></div>
                  <div>
                    <span className="font-semibold text-blue-600">Programming Language:</span>
                    <span className="text-gray-700"> Python was used to build the core machine learning model.</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-purple-600 rounded-full mt-2"></div>
                  <div>
                    <span className="font-semibold text-purple-600">Machine Learning Library:</span>
                    <span className="text-gray-700"> Scikit-learn was implemented for building and training the linear regression model.</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-600 rounded-full mt-2"></div>
                  <div>
                    <span className="font-semibold text-green-600">Dataset:</span>
                    <span className="text-gray-700"> A historical home dataset was used to train and evaluate the model.</span>
                  </div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-orange-600 rounded-full mt-2"></div>
                  <div>
                    <span className="font-semibold text-orange-600">Development Platforms:</span>
                    <span className="text-gray-700"> VS Code and Replit were used for coding, testing, and collaboration.</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-pink-600 rounded-full mt-2"></div>
                  <div>
                    <span className="font-semibold text-pink-600">Frontend Technologies:</span>
                    <span className="text-gray-700"> HTML, CSS, JavaScript, and React were used to design a responsive and interactive user interface.</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-3">
              <Database className="h-6 w-6 text-green-600" />
              <span className="text-green-600">The Data</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700 mb-6 leading-relaxed">
              We source our data from multiple reliable real estate databases, ensuring a comprehensive and accurate foundation for our prediction model. Our dataset includes:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <Brain className="h-5 w-5 text-blue-600" />
                  <span className="text-gray-700">Historical home sales data</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Home className="h-5 w-5 text-purple-600" />
                  <span className="text-gray-700">Property characteristics (size, bedrooms, bathrooms, etc.)</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <Globe className="h-5 w-5 text-green-600" />
                  <span className="text-gray-700">Location information and neighborhood statistics</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Code className="h-5 w-5 text-orange-600" />
                  <span className="text-gray-700">Market trends and economic indicators</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Credits Footer */}
      <footer className="bg-gradient-to-r from-blue-50 to-purple-50 border-t border-gray-200 py-6 mt-12">
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