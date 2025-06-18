import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { ModelMetrics } from "@shared/schema";

export default function ModelMetrics() {
  const { data: metrics, isLoading, error } = useQuery<ModelMetrics>({
    queryKey: ['/api/model-metrics'],
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <BarChart3 className="h-5 w-5 text-purple-600" />
            Model Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="flex justify-between items-center">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-12" />
              </div>
            ))}
            <div className="pt-2 border-t border-gray-100">
              <Skeleton className="h-3 w-32 mx-auto" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !metrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <BarChart3 className="h-5 w-5 text-purple-600" />
            Model Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-red-500 text-sm">
            Failed to load model metrics
          </div>
        </CardContent>
      </Card>
    );
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-3">
          <BarChart3 className="h-5 w-5 text-purple-600" />
          Model Performance
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">R² Score</span>
            <span className="text-sm font-semibold text-gray-900">{metrics.rSquared}</span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">RMSE</span>
            <span className="text-sm font-semibold text-gray-900">{formatCurrency(metrics.rmse)}</span>
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">MAE</span>
            <span className="text-sm font-semibold text-gray-900">{formatCurrency(metrics.mae)}</span>
          </div>
          
          <div className="pt-2 border-t border-gray-100">
            <div className="text-xs text-gray-500 text-center">
              Last trained: {metrics.lastTrained}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
