import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Edit, Calculator } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { insertPropertySchema, type PredictionRequest, type PredictionResponse } from "@shared/schema";
import { z } from "zod";

const formSchema = insertPropertySchema.extend({
  squareFootage: z.number().min(500, "Square footage must be at least 500").max(10000, "Square footage must be less than 10,000"),
  yearBuilt: z.number().min(1900, "Year built must be after 1900").max(2024, "Year built cannot be in the future"),
  bedrooms: z.number().min(1, "Must have at least 1 bedroom").max(10, "Too many bedrooms"),
  bathrooms: z.number().min(1, "Must have at least 1 bathroom").max(10, "Too many bathrooms"),
});

export default function PropertyForm() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      squareFootage: 2200,
      yearBuilt: 2005,
      bedrooms: 3,
      bathrooms: 2,
      garage: 2,
      propertyType: "single-family",
      neighborhood: "suburbs",
      hasPool: false,
      hasFireplace: true,
      hasHardwoodFloors: true,
      recentlyUpdated: false,
    },
  });

  const predictMutation = useMutation({
    mutationFn: async (data: PredictionRequest): Promise<PredictionResponse> => {
      const response = await apiRequest("POST", "/api/predict", data);
      return await response.json();
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['prediction'], data);
      toast({
        title: "Prediction Complete",
        description: `Estimated price: ${new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(data.estimatedPrice)}`,
      });
    },
    onError: (error) => {
      toast({
        title: "Prediction Failed",
        description: error instanceof Error ? error.message : "Unable to generate prediction",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (values: z.infer<typeof formSchema>) => {
    const predictionRequest: PredictionRequest = {
      squareFootage: values.squareFootage,
      yearBuilt: values.yearBuilt,
      bedrooms: values.bedrooms,
      bathrooms: values.bathrooms,
      garage: values.garage ?? 0,
      propertyType: values.propertyType,
      neighborhood: values.neighborhood,
      hasPool: values.hasPool ?? false,
      hasFireplace: values.hasFireplace ?? false,
      hasHardwoodFloors: values.hasHardwoodFloors ?? false,
      recentlyUpdated: values.recentlyUpdated ?? false,
    };
    
    predictMutation.mutate(predictionRequest);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-3">
          <Edit className="h-5 w-5 text-primary" />
          Property Details
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            {/* Property Basic Info */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FormField
                control={form.control}
                name="squareFootage"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Square Footage</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Input
                          type="number"
                          placeholder="e.g., 2000"
                          {...field}
                          onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                        />
                        <div className="absolute right-3 top-3 text-gray-400 text-sm">sq ft</div>
                      </div>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="yearBuilt"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Year Built</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        placeholder="e.g., 1995"
                        {...field}
                        onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Room Configuration */}
            <div className="grid grid-cols-3 gap-4">
              <FormField
                control={form.control}
                name="bedrooms"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Bedrooms</FormLabel>
                    <Select onValueChange={(value) => field.onChange(parseInt(value))} defaultValue={field.value?.toString()}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="1">1</SelectItem>
                        <SelectItem value="2">2</SelectItem>
                        <SelectItem value="3">3</SelectItem>
                        <SelectItem value="4">4</SelectItem>
                        <SelectItem value="5">5+</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="bathrooms"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Bathrooms</FormLabel>
                    <Select onValueChange={(value) => field.onChange(parseFloat(value))} defaultValue={field.value?.toString()}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="1">1</SelectItem>
                        <SelectItem value="1.5">1.5</SelectItem>
                        <SelectItem value="2">2</SelectItem>
                        <SelectItem value="2.5">2.5</SelectItem>
                        <SelectItem value="3">3</SelectItem>
                        <SelectItem value="3.5">3.5</SelectItem>
                        <SelectItem value="4">4+</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="garage"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Garage</FormLabel>
                    <Select onValueChange={(value) => field.onChange(parseInt(value))} defaultValue={field.value?.toString()}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="0">None</SelectItem>
                        <SelectItem value="1">1 Car</SelectItem>
                        <SelectItem value="2">2 Car</SelectItem>
                        <SelectItem value="3">3+ Car</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Property Type */}
            <div className="grid grid-cols-1 gap-6">
              <FormField
                control={form.control}
                name="propertyType"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Property Type</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select Type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="single-family">Single Family</SelectItem>
                        <SelectItem value="condo">Condominium</SelectItem>
                        <SelectItem value="townhouse">Townhouse</SelectItem>
                        <SelectItem value="multi-family">Multi-Family</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Submit Button */}
            <div className="pt-4">
              <Button 
                type="submit" 
                className="w-full"
                disabled={predictMutation.isPending}
              >
                <Calculator className="mr-2 h-4 w-4" />
                {predictMutation.isPending ? "Calculating..." : "Get Price Prediction"}
              </Button>
            </div>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
