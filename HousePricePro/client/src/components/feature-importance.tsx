import { useQuery } from "@tanstack/react-query";
import { Weight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { PredictionResponse } from "@shared/schema";

const colorMap: Record<number, string> = {
  0: "bg-primary",
  1: "bg-secondary", 
  2: "bg-orange-500",
  3: "bg-gray-400",
  4: "bg-purple-500",
};

export default function FeatureImportance() {
  const { data: prediction, isLoading } = useQuery<PredictionResponse | null>({
    queryKey: ['prediction'],
    initialData: null,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <Weight className="h-5 w-5 text-orange-500" />
            Key Price Factors
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Skeleton className="w-3 h-3 rounded-full" />
                  <Skeleton className="h-4 w-24" />
                </div>
                <Skeleton className="h-4 w-8" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction?.featureImportance) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3">
            <Weight className="h-5 w-5 text-orange-500" />
            Key Price Factors
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-500 text-sm">
            Feature importance will appear after prediction
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-3">
          <Weight className="h-5 w-5 text-orange-500" />
          Key Price Factors
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {prediction.featureImportance.map((item, index) => (
            <div key={item.feature} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 ${colorMap[index] || 'bg-gray-400'} rounded-full`}></div>
                <span className="text-sm text-gray-700">{item.feature}</span>
              </div>
              <div className="text-sm font-medium text-gray-900">{item.importance}%</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
