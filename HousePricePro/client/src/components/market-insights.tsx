import { TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function MarketInsights() {
  // Mock market data for demonstration
  const marketStats = {
    avgPrice: "$425K",
    priceChange: "+5.2%",
    daysOnMarket: "32",
    inventory: "2.1"
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-3">
          <TrendingUp className="h-6 w-6 text-primary" />
          Market Insights & Trends
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Price Trend Chart Placeholder */}
          <div className="space-y-4">
            <h4 className="text-lg font-medium text-gray-900">Regional Price Trends</h4>
            <div className="h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <TrendingUp className="h-12 w-12 mx-auto mb-2" />
                <p className="text-sm">Interactive price trend chart</p>
                <p className="text-xs">Chart.js integration coming soon</p>
              </div>
            </div>
          </div>
          
          {/* Feature Distribution */}
          <div className="space-y-4">
            <h4 className="text-lg font-medium text-gray-900">Property Feature Analysis</h4>
            <div className="h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <TrendingUp className="h-12 w-12 mx-auto mb-2" />
                <p className="text-sm">Feature distribution charts</p>
                <p className="text-xs">Interactive visualizations</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Market Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-primary">{marketStats.avgPrice}</div>
            <div className="text-sm text-gray-600">Avg. Home Price</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-secondary">{marketStats.priceChange}</div>
            <div className="text-sm text-gray-600">YoY Price Change</div>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">{marketStats.daysOnMarket}</div>
            <div className="text-sm text-gray-600">Avg. Days on Market</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{marketStats.inventory}</div>
            <div className="text-sm text-gray-600">Months of Inventory</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
