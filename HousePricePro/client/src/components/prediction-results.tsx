import { useQuery } from "@tanstack/react-query";
import { TrendingUp, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { PredictionResponse } from "@shared/schema";

export default function PredictionResults() {
  const { data: prediction, isLoading } = useQuery<PredictionResponse | null>({
    queryKey: ['prediction'],
    initialData: null,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <TrendingUp className="h-5 w-5 text-secondary" />
            Price Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center space-y-4">
            <Skeleton className="h-12 w-32 mx-auto" />
            <Skeleton className="h-4 w-24 mx-auto" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <TrendingUp className="h-5 w-5 text-secondary" />
            Price Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-500">
            <div className="text-2xl font-bold text-gray-300 mb-2">$---,---</div>
            <div className="text-sm mb-4">Enter property details to get prediction</div>
            <div className="text-xs">
              <Info className="inline h-3 w-3 mr-1" />
              Prediction will appear here
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-3">
          <TrendingUp className="h-5 w-5 text-secondary" />
          Price Prediction
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center">
          <div className="text-4xl font-bold text-primary mb-2">
            {formatCurrency(prediction.estimatedPrice)}
          </div>
          <div className="text-sm text-gray-600 mb-4">Estimated Market Value</div>
          
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Confidence Range</span>
              <span className="text-sm font-medium text-secondary">
                {Math.round(prediction.confidence * 100)}%
              </span>
            </div>
            <div className="text-sm text-gray-700">
              {formatCurrency(prediction.lowerBound)} - {formatCurrency(prediction.upperBound)}
            </div>
          </div>
          
          <div className="text-xs text-gray-500">
            <Info className="inline h-3 w-3 mr-1" />
            Prediction based on current market data
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
